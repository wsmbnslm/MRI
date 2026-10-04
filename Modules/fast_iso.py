"""Vectorized drop-in replacement for ``MRzeroCore.isochromat_sim``.

Why the original is slow
------------------------
The pypulseq importer splits each 96-sample readout into one event per ADC
sample, so a typical repetition has ~120 events.  The original simulation
loops over every event of every repetition (~760k events for 6400 spokes)
and, per event:

* clones the full ``(voxels, spins, 3)`` magnetization tensor,
* applies T1/T2 relaxation into a fresh ``empty_like`` tensor,
* builds a full 3x3 rotation matrix of shape ``(voxels*spins, 3, 3)`` for the
  T2' dephasing (zero-initialized, then filled with 5 indexed writes), and
  applies it with ``einsum``.

For a 96^3 phantom with 64 spins that is several GB of memory traffic per
event, i.e. petabytes total - hence hours of runtime.

This version exploits the fact that all four per-event rotations (T2'
dephasing, intra-voxel precession, B0 precession and gradient precession)
rotate around the same z-axis, so they combine into a single rotation per
(voxel, spin, event).  The magnetization is kept as a complex transverse
tensor ``w = Mx + iMy`` and, for a chunk of events, the rotated states are
computed as ``w * exp(1j * angle) * exp(-t / T2)`` in a handful of
elementwise kernels.  All events of a repetition are handled in this way;
only the last event's state is carried into the next repetition.  The math
is the same as the original (up to floating point re-ordering).
"""
from __future__ import annotations

import torch

from MRzeroCore.sequence import PulseUsage


def _flip(spins: torch.Tensor, angle: torch.Tensor, phase: torch.Tensor,
          B1: torch.Tensor) -> torch.Tensor:
    """Rz(phase) Rx(angle * B1) Rz(-phase), elementwise instead of einsum."""
    a = (angle * B1).to(spins.dtype)
    p = torch.as_tensor(phase, device=spins.device, dtype=spins.dtype)

    sp, cp = torch.sin(p), torch.cos(p)
    sa, ca = torch.sin(a), torch.cos(a)

    if a.dim() > 0:
        sa, ca = sa.unsqueeze(-1), ca.unsqueeze(-1)
    if p.dim() > 0 and p.numel() == B1.numel():
        sp, cp = sp.unsqueeze(-1), cp.unsqueeze(-1)

    sp2, cp2 = sp * sp, cp * cp
    o12 = (1 - ca) * sp * cp

    out = torch.empty_like(spins)
    x, y, z = spins[..., 0], spins[..., 1], spins[..., 2]
    out[..., 0] = (sp2 * ca + cp2) * x + o12 * y + (sa * sp) * z
    out[..., 1] = o12 * x + (sp2 + ca * cp2) * y - (sa * cp) * z
    out[..., 2] = -(sa * sp) * x + (sa * cp) * y + ca * z
    return out


def isochromat_sim_fast(seq, data, spin_count: int,
                        perfect_spoiling: bool = False,
                        print_progress: bool = True,
                        spin_dist: str = "rand",
                        r2_seed: torch.Tensor | None = None,
                        chunk_bytes: int = 8 * 2**30) -> torch.Tensor:
    """Same signature and return value as ``MRzeroCore.isochromat_sim``.

    Parameters
    ----------
    chunk_bytes : int
        Memory budget for the temporary ``events x voxels x spins`` tensors;
        the number of events processed per chunk is derived from it.
    """
    from numpy import pi

    device = data.device

    voxel_size = 0.5 / torch.tensor(data.nyquist, device=device)
    if not torch.isfinite(voxel_size).all():
        voxel_size = torch.tensor([0.1, 0.1, 0.1], device=device)

    if spin_dist == "rand":
        spin_pos = torch.rand(spin_count, 3, device=device)
    elif spin_dist == "r2":
        if r2_seed is None:
            r2_seed = torch.rand(3, device=device)
        g = 1.22074408460575947536
        a = 1.0 / torch.tensor([g**1, g**2, g**3], device=device)
        indices = torch.arange(spin_count, device=device)
        spin_pos = torch.stack([
            (r2_seed[0] + a[0] * indices) % 1,
            (r2_seed[1] + a[1] * indices) % 1,
            (r2_seed[2] + a[2] * indices) % 1,
        ], dim=1)
    else:
        raise ValueError("unexpected spin_dist", spin_dist)

    spin_pos = 2 * pi * (spin_pos - 0.5) * voxel_size.unsqueeze(0)

    off_res = torch.linspace(-0.5, 0.5, spin_count, device=device)
    omega = torch.tan(pi * 0.999 * off_res)
    omega = omega[torch.randperm(spin_count)]

    coil_sensitivity = (
        data.coil_sens.t().to(torch.cfloat)
        * data.PD.unsqueeze(1) / spin_count
    )
    coil_count = data.coil_sens.shape[0]
    V = data.PD.numel()
    S = spin_count

    # (voxel, spin) part of the total z-rotation angle: T2' dephasing + B0
    base = (1 / data.T2dash).unsqueeze(1) * omega.unsqueeze(0) \
        - (2 * pi * data.B0).unsqueeze(1)
    two_pi_vp = 2 * pi * data.voxel_pos
    inv_t2 = 1 / data.T2
    inv_t1 = 1 / data.T1

    # ~12 bytes of temporaries per (event, voxel, spin)
    chunk = max(4, min(64, int(chunk_bytes / (12 * V * S))))

    signal_parts = []

    spins = torch.zeros((V, S, 3), device=device)
    spins[:, :, 2] = 1

    for r, rep in enumerate(seq):
        if print_progress:
            print(f"\r {r+1} / {len(seq)}", end="")

        if perfect_spoiling and rep.pulse.usage == PulseUsage.EXCIT:
            spins[..., :2].zero_()

        spins = _flip(spins, rep.pulse.angle, rep.pulse.phase, data.B1)

        t = rep.event_time.cumsum(0).to(device)          # (E,)
        gm = rep.gradm.cumsum(0).to(device)              # (E, 3)
        E = rep.event_count
        adc = rep.adc_usage > 0

        w = spins[..., 0] + 1j * spins[..., 1]           # (V, S) complex
        z = spins[..., 2]

        for e0 in range(0, E, chunk):
            e1 = min(E, e0 + chunk)
            C = e1 - e0
            tc, gmc = t[e0:e1], gm[e0:e1]

            ang = base.unsqueeze(0) * tc.view(C, 1, 1)
            ang -= (gmc @ spin_pos.t()).reshape(C, 1, S)
            ang -= (two_pi_vp @ gmc.t()).t().reshape(C, V, 1)

            u = (ang * 1j).exp_()                        # (C, V, S) complex
            del ang
            u.mul_(w.view(1, V, S))
            u.mul_(torch.exp(-tc.view(C, 1) * inv_t2.view(1, V)).unsqueeze(-1))

            if adc[e0:e1].any():
                sig = u[adc[e0:e1]].sum(2) @ coil_sensitivity
                rot = torch.exp(1j * rep.adc_phase[e0:e1][adc[e0:e1]]
                                .to(torch.cfloat))
                signal_parts.append(sig * rot.view(-1, 1))

            if e1 == E:
                spins = torch.empty_like(spins)
                spins[..., 0] = u[-1].real
                spins[..., 1] = u[-1].imag
                r1 = torch.exp(-t[-1] * inv_t1).unsqueeze(1)
                spins[..., 2] = z * r1 + (1 - r1)

        del u

    if print_progress:
        print(" - done")
    return torch.cat(signal_parts)
