#!/usr/bin/env python3
"""Correctness + performance benchmark of ``isochromat_sim_fast`` (the
vectorized drop-in in ``examples/Modules/fast_iso.py``) against the reference
implementation ``MRzeroCore.isochromat_sim``.

Evidence produced (all written to ``reports/``):
  1. Equivalence   -- same seed / seq / data, original vs fast: max abs error,
                      relative L2 error, complex correlation.
  2. Naive ref     -- an independently written per-event Bloch-equation loop
                      (explicit 3-D rotations, no einsum, no chunking) compared
                      against the fast implementation at a small scale.
  3. Physics check -- inversion-recovery null: a single-tissue phantom must
                      null at TI = T1 * ln(2), independent of both
                      implementations.
  3b. Bug report  -- the ORIGINAL implementation crashes on
                      spin_dist="r2" (shape bug, shape (3,S) vs (S,3));
                      minimal repro recorded, plus a demonstration that the
                      fast implementation implements the documented (S,3)
                      behaviour and runs fine with r2.
  4. Performance   -- head-to-head timing + peak memory at 8^3..64^3, a
                      scaling plot, a 128^3 production-scale fast-only run,
                      and an extrapolation for the full 128^3 / 6400-readout
                      configuration (clearly labelled as extrapolated).

Usage:
    python benchmarks/bench_fast_iso.py            # full benchmark (~50 min)
    python benchmarks/bench_fast_iso.py --quick    # small cases only (pilot)

Outputs:
    reports/fast_iso_benchmark.json
    reports/equivalence_*.png
    reports/naive_reference.png
    reports/ir_null_curve.png
    reports/scaling_time.png
    reports/scaling_memory.png
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import pypulseq as pp
import MRzeroCore as mr0

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
REPORTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "examples"))
from Modules.system import default_system            # noqa: E402
from Modules.fast_iso import isochromat_sim_fast     # noqa: E402

SEED = 20261019
DEVICE = torch.device("cuda")
# MRzeroCore's VoxelGridPhantom.build() creates internal tensors with
# torch.tensor(...), i.e. on the default device - the thesis notebooks set
# this to cuda for exactly that reason.
torch.set_default_device(DEVICE)


# --------------------------------------------------------------------------
# environment / provenance
# --------------------------------------------------------------------------
def env_info() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, cwd=REPO).stdout.strip()
        except Exception:
            return "?"
    try:
        import MRzeroCore
        mr0_ver = getattr(MRzeroCore, "__version__", "unknown")
    except Exception:
        mr0_ver = "unknown"
    return {
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "git_repo": run("git rev-parse --show-toplevel"),
        "git_head": run("git rev-parse HEAD"),
        "fast_iso_blame": run("git log -1 --format=%h -- examples/Modules/fast_iso.py"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "mr0": mr0_ver,
        "seed": SEED,
    }


# --------------------------------------------------------------------------
# synthetic phantom (self-contained, deterministic; no external data files)
# --------------------------------------------------------------------------
def build_phantom(n: int, seed: int, coil_count: int = 4,
                  single_tissue: tuple | None = None) -> mr0.SimData:
    """3-tissue spherical phantom (core / shell / background), or a uniform
    single-tissue phantom when ``single_tissue=(T1, T2, PD)`` is given."""
    torch.manual_seed(seed)
    grid = torch.linspace(-0.5, 0.5, n)
    gx, gy, gz = torch.meshgrid(grid, grid, grid, indexing="ij")
    r = (gx ** 2 + gy ** 2 + gz ** 2).sqrt()
    tissue = torch.where(r < 0.15, 2, torch.where(r < 0.4, 1, 0)).to(torch.int32)
    if single_tissue is not None:
        t1, t2, pd = single_tissue
        pd_map = torch.full((n, n, n), pd)
        t1_map = torch.full((n, n, n), t1)
        t2_map = torch.full((n, n, n), t2)
    else:
        pd_map = torch.where(tissue == 2, 1.0,
                             torch.where(tissue == 1, 0.8, 0.7))
        t1_map = torch.where(tissue == 2, 1.3,
                             torch.where(tissue == 1, 0.9, 4.0))
        t2_map = torch.where(tissue == 2, 0.08,
                             torch.where(tissue == 1, 0.07, 0.5))
    t2dash_map = torch.full_like(pd_map, 0.04)
    b0_map = torch.zeros_like(pd_map)
    b1_map = torch.ones_like(pd_map).unsqueeze(0)

    fov = 0.25
    shape = torch.tensor(pd_map.shape, dtype=torch.float32)
    dv = fov * 1000.0 / shape
    affine = torch.eye(4, device=DEVICE)
    for i in range(3):
        affine[i, i] = dv[i]
        affine[i, 3] = -dv[i] * (shape[i] // 2)

    phantom = mr0.VoxelGridPhantom(
        pd_map.to(DEVICE), t1_map.to(DEVICE), t2_map.to(DEVICE),
        t2dash_map.to(DEVICE), b0_map.to(DEVICE), b0_map.to(DEVICE),
        b1_map.to(DEVICE),
        torch.ones(1, *pd_map.shape, device=DEVICE), affine,
    )

    # Smooth single-peaked complex coil sensitivities (deterministic).
    coil_sens = torch.zeros(coil_count, *pd_map.shape, dtype=torch.cfloat,
                            device=DEVICE)
    for c in range(coil_count):
        cx = 0.3 * math.cos(2 * math.pi * c / coil_count)
        cy = 0.3 * math.sin(2 * math.pi * c / coil_count)
        gz_c = 0.2 * (c - coil_count / 2)
        d2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - gz_c) ** 2
        coil_sens[c] = torch.exp(-d2 / 0.08).to(torch.cfloat)
    coil_sens = coil_sens / coil_sens.abs().amax()
    phantom.coil_sens = coil_sens
    phantom.B0 = 0 * phantom.B0
    phantom.B1 = torch.ones(phantom.B1.shape, dtype=phantom.B1.dtype)
    return phantom.build()


# --------------------------------------------------------------------------
# sequence builder (slab-selective 60-degree bSSFP, golden-angle radial
# readout, optional leading 180-degree inversion -- same construction as
# examples/3D_IR_TrueFISP_bssfp.ipynb, without the kz partition encoding)
# --------------------------------------------------------------------------
def build_bssfp_seq(n_x: int, n_readouts: int, inv_dur: float = 2e-3,
                    ti_start: float | None = 20e-3) -> pp.Sequence:
    system = default_system()
    fov = 0.25
    seq = pp.Sequence(system)
    deltak = 1 / fov
    _gr = seq.grad_raster_time
    ga_deg = 111.246

    rf, gz, _ = pp.make_sinc_pulse(
        apodization=0.5, duration=0.8e-3,
        flip_angle=np.deg2rad(60.0), slice_thickness=0.2,
        system=system, time_bw_product=2.0, return_gz=True, use="excitation")

    _astep = n_x * seq.adc_raster_time
    _k = 1
    while abs(_k * _gr - round(_k * _gr / _astep) * _astep) > 1e-9:
        _k += 1
    readout_flat = int(round((1.0 / 890.0) / (_k * _gr))) * (_k * _gr)
    gx_base = pp.make_trapezoid(
        channel="x", flat_area=n_x * deltak, flat_time=readout_flat,
        system=system)
    adc = pp.make_adc(num_samples=n_x, duration=gx_base.flat_time,
                      delay=gx_base.rise_time, system=system)

    min_x = pp.calc_duration(
        pp.make_trapezoid(channel="x", area=abs(gx_base.area) / 2,
                          system=system))
    min_z = pp.calc_duration(
        pp.make_trapezoid(channel="z", area=abs(gz.area) / 2, system=system))
    pre_dur = int(np.ceil(max(0.6e-3, min_x, min_z) / _gr - 1e-9)) * _gr
    gx_pre_base = pp.make_trapezoid(
        channel="x", area=-gx_base.area / 2, duration=pre_dur, system=system)
    gz_pre = pp.make_trapezoid(
        channel="z", area=-gz.area / 2, duration=pre_dur, system=system)

    tr = 3.57e-3
    dur_exc = pp.calc_duration(rf, gz)
    dur_ro = pp.calc_duration(gx_base)
    free = tr - dur_exc - 2 * pre_dur - dur_ro
    delay_te = int(np.ceil(max(0.0, free) / _gr - 1e-9)) * _gr

    if ti_start is not None:
        rf_inv = pp.make_block_pulse(flip_angle=np.deg2rad(180),
                                     duration=inv_dur, system=system,
                                     use="inversion")
        inv_duration = pp.calc_duration(rf_inv)
        delay_ti = np.ceil((ti_start - inv_duration / 2 - dur_exc / 2) / _gr) * _gr
        assert delay_ti > 0, "ti_start too short"
        seq.add_block(rf_inv)
        seq.add_block(pp.make_delay(delay_ti))

    rf_phase = 0.0
    for s in range(n_readouts):
        angle = np.deg2rad(s * ga_deg)
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_phase = (rf_phase + 180.0) % 360
        kx, ky = np.cos(angle), np.sin(angle)

        seq.add_block(rf, gz)
        seq.add_block(
            pp.make_trapezoid(channel="x", area=gx_pre_base.area * kx,
                              duration=pre_dur, system=system),
            pp.make_trapezoid(channel="y", area=gx_pre_base.area * ky,
                              duration=pre_dur, system=system),
            gz_pre,
        )
        if delay_te > 0:
            seq.add_block(pp.make_delay(delay_te))
        gx = pp.make_trapezoid(
            channel="x", amplitude=gx_base.amplitude * kx,
            flat_time=gx_base.flat_time, rise_time=gx_base.rise_time,
            fall_time=gx_base.fall_time, system=system)
        gy = pp.make_trapezoid(
            channel="y", amplitude=gx_base.amplitude * ky,
            flat_time=gx_base.flat_time, rise_time=gx_base.rise_time,
            fall_time=gx_base.fall_time, system=system)
        seq.add_block(gx, gy, adc)
        seq.add_block(
            pp.make_trapezoid(channel="x", area=gx_pre_base.area * kx,
                              duration=pre_dur, system=system),
            pp.make_trapezoid(channel="y", area=gx_pre_base.area * ky,
                              duration=pre_dur, system=system),
            gz_pre,
        )

    ok, err = seq.check_timing()
    if not ok:
        raise RuntimeError(f"sequence timing failed: {err}")
    seq.set_definition("FOV", [fov, fov, 0.25])
    return seq


def to_mr0_seq(seq: pp.Sequence, name: str) -> mr0.Sequence:
    """Write + re-import, exactly like the thesis notebooks do."""
    path = REPORTS / name
    seq.write(str(path))
    return mr0.Sequence.import_file(str(path))


def seq_stats(seq: mr0.Sequence) -> dict:
    n_events = sum(r.event_count for r in seq)
    n_adc = sum(int((r.adc_usage > 0).sum()) for r in seq)
    n_reps = len(seq)
    return {"n_reps": n_reps, "n_events": int(n_events), "n_adc": int(n_adc)}


# --------------------------------------------------------------------------
# spin distribution (replicates the RNG draw order of BOTH implementations:
# torch.rand(S,3) [or torch.rand(3) for r2] first, then torch.randperm(S))
# --------------------------------------------------------------------------
def spin_distribution(spin_count: int, spin_dist: str, voxel_size: torch.Tensor,
                      r2_seed: torch.Tensor | None = None):
    if spin_dist == "rand":
        spin_pos = torch.rand(spin_count, 3, device=DEVICE)
    elif spin_dist == "r2":
        if r2_seed is None:
            r2_seed = torch.rand(3, device=DEVICE)
        g = 1.22074408460575947536
        a = 1.0 / torch.tensor([g ** 1, g ** 2, g ** 3], device=DEVICE)
        indices = torch.arange(spin_count, device=DEVICE)
        spin_pos = torch.stack([
            (r2_seed[0] + a[0] * indices) % 1,
            (r2_seed[1] + a[1] * indices) % 1,
            (r2_seed[2] + a[2] * indices) % 1,
        ], dim=1)
    spin_pos = 2 * np.pi * (spin_pos - 0.5) * voxel_size.unsqueeze(0)
    off_res = torch.linspace(-0.5, 0.5, spin_count, device=DEVICE)
    omega = torch.tan(np.pi * 0.999 * off_res)
    omega = omega[torch.randperm(spin_count)]
    return spin_pos, omega, r2_seed


# --------------------------------------------------------------------------
# independent naive reference: per-event Bloch dynamics written out with
# explicit 3-D rotation algebra (no einsum, no chunking, no matrix assembly)
# --------------------------------------------------------------------------
def naive_sim(seq: mr0.Sequence, data: mr0.SimData, spin_count: int,
              spin_dist: str, r2_seed: torch.Tensor | None = None) -> torch.Tensor:
    voxel_size = 0.5 / torch.tensor(data.nyquist, device=DEVICE)
    if not torch.isfinite(voxel_size).all():
        voxel_size = torch.tensor([0.1, 0.1, 0.1], device=DEVICE)
    spin_pos, omega, _ = spin_distribution(spin_count, spin_dist, voxel_size,
                                           r2_seed)
    coil_sens = (data.coil_sens.t().to(torch.cfloat)
                 * data.PD.unsqueeze(1) / spin_count)
    v = data.PD.numel()

    def rz(x, y, ang):
        c, s = torch.cos(ang), torch.sin(ang)
        return x * c - y * s, x * s + y * c

    def rx(y, z, ang):
        c, s = torch.cos(ang), torch.sin(ang)
        return y * c - z * s, y * s + z * c

    spins = torch.zeros((v, spin_count, 3), device=DEVICE)
    spins[:, :, 2] = 1.0
    signal = []

    for rep in seq:
        # RF flip: Rz(p) Rx(a) Rz(-p), applied as three explicit rotations
        x, y, z = spins[..., 0], spins[..., 1], spins[..., 2]
        b1 = data.B1.squeeze(0) if data.B1.dim() == 2 else data.B1
        a = (rep.pulse.angle * b1).view(-1, 1)   # (V,1) vs spins (V,S)
        p = torch.as_tensor(rep.pulse.phase, device=DEVICE)
        x, y = rz(x, y, -p)
        y, z = rx(y, z, a)
        x, y = rz(x, y, p)
        spins = torch.stack((x, y, z), dim=-1)

        t = rep.event_time.cumsum(0).to(DEVICE)
        gm = rep.gradm.cumsum(0).to(DEVICE)
        e_last = rep.event_count - 1
        for e in range(rep.event_count):
            te = t[e]
            gm_e = gm[e]
            # relax (from the state right after the pulse, like the original)
            e1 = torch.exp(-te / data.T1)
            e2 = torch.exp(-te / data.T2)
            x = spins[..., 0] * e2.view(-1, 1)
            y = spins[..., 1] * e2.view(-1, 1)
            z = spins[..., 2] * e1.view(-1, 1) + (1 - e1).view(-1, 1)
            # four z-rotations, applied one by one (same signs as original)
            x2, y2 = rz(x, y, (1 / data.T2dash).unsqueeze(1) * omega.unsqueeze(0) * te)
            x2, y2 = rz(x2, y2, -(spin_pos @ gm_e).unsqueeze(0))
            x2, y2 = rz(x2, y2, (-2 * np.pi * data.B0 * te).unsqueeze(1))
            x2, y2 = rz(x2, y2, (-2 * np.pi * (data.voxel_pos @ gm_e)).unsqueeze(1))
            x, y = x2, y2
            if rep.adc_usage[e] > 0:
                mag = x.sum(1) + 1j * y.sum(1)
                sig = (mag @ coil_sens) * torch.exp(
                    1j * rep.adc_phase[e].to(DEVICE))
                signal.append(sig)
            if e == e_last:
                spins = torch.stack((x, y, z), dim=-1)
    return torch.cat([s.unsqueeze(0) for s in signal])


# --------------------------------------------------------------------------
# metrics / timing helpers
# --------------------------------------------------------------------------
def signal_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    assert a.shape == b.shape, (a.shape, b.shape)
    a, b = a.flatten(), b.flatten()
    d = (a - b).abs()
    rel = (a - b).norm() / a.norm().clamp_min(1e-30)
    corr = torch.vdot(a, b).abs() / (a.norm() * b.norm()).clamp_min(1e-30)
    return {
        "shape": list(a.shape),
        "max_abs_error": float(d.max()),
        "rel_l2_error": float(rel),
        "correlation": float(corr.real),
        "allclose_rtol1e-2_atol1e-3": bool(torch.allclose(a, b, rtol=1e-2, atol=1e-3)),
        "max_abs_of_ref": float(a.abs().max()),
    }


def run_timed(fn, iters: int, warmup: int = 0):
    """Median wall time [s] and peak allocation delta [GiB]."""
    times, mems = [], []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        mems.append((torch.cuda.max_memory_allocated() - base) / 2 ** 30)
    return float(statistics.median(times)), float(statistics.median(mems)), out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="run small cases only (pilot)")
    args = ap.parse_args()

    out = {"env": env_info()}
    print(json.dumps(out["env"], indent=2))

    # ---------------- 1 + 2: equivalence (incl. naive reference) ----------
    if args.quick:
        equiv_cases = [dict(n=16, K=40, S=64, spin_dist="rand", seed=0)]
    else:
        # Note: spin_dist="r2" is NOT usable for equivalence: the ORIGINAL
        # implementation crashes on it (shape bug in its r2 branch, see
        # "original_r2_bug" below). Two rand cases at different spin counts
        # instead.
        equiv_cases = [
            dict(n=16, K=40, S=64, spin_dist="rand", seed=0),
            dict(n=16, K=40, S=16, spin_dist="rand", seed=1),
            dict(n=32, K=20, S=64, spin_dist="rand", seed=0),
        ]
    out["equivalence"] = []
    for case in equiv_cases:
        print(f"\n=== equivalence n={case['n']} K={case['K']} dist={case['spin_dist']} ===")
        data = build_phantom(case["n"], seed=1000 + case["n"])
        seq = to_mr0_seq(build_bssfp_seq(case["n"], case["K"]),
                         f"bench_n{case['n']}.seq")
        st = seq_stats(seq)

        torch.manual_seed(case["seed"])
        s_ref = mr0.isochromat_sim(seq, data, case["S"], print_progress=False,
                                   spin_dist=case["spin_dist"])
        torch.manual_seed(case["seed"])
        s_fast = isochromat_sim_fast(seq, data, case["S"],
                                     print_progress=False,
                                     spin_dist=case["spin_dist"])
        m = signal_metrics(s_ref, s_fast)
        m.update({"case": {k: case[k] for k in ("n", "K", "S", "spin_dist")}, **st})
        out["equivalence"].append(m)
        print(m)
        if case["n"] == 16 and case["spin_dist"] == "rand":
            fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            n = min(400, s_ref.shape[0])
            ax[0].plot(s_ref[:n, 0].real.cpu(), label="reference", lw=1)
            ax[0].plot(s_fast[:n, 0].real.cpu(), label="fast", lw=1, alpha=0.8)
            ax[0].set_ylabel("Re signal (coil 0)")
            ax[0].legend()
            ax[1].plot(((s_ref - s_fast)[:n, 0]).abs().cpu(), lw=1, color="tab:red")
            ax[1].set_ylabel("|ref - fast|")
            ax[1].set_xlabel("sample")
            fig.tight_layout()
            fig.savefig(REPORTS / "equivalence_n16.png", dpi=150)
            plt.close(fig)

    # ---------------- original spin_dist="r2" bug (minimal repro) ---------
    print("\n=== original spin_dist='r2' branch: minimal repro ===")
    data = build_phantom(4, seed=1004)
    seq = to_mr0_seq(build_bssfp_seq(4, 2, ti_start=30e-3), "bench_r2bug.seq")
    r2_bug = {}
    for name, fn in [
        ("original", lambda: mr0.isochromat_sim(
            seq, data, 64, print_progress=False, spin_dist="r2")),
        ("fast", lambda: isochromat_sim_fast(
            seq, data, 64, print_progress=False, spin_dist="r2")),
    ]:
        torch.manual_seed(0)
        try:
            res = fn()
            r2_bug[name] = {"status": "ran", "shape": list(res.shape)}
        except Exception as e:
            r2_bug[name] = {"status": "crash",
                            "error": f"{type(e).__name__}: {e}"}
        print(name, "->", r2_bug[name])
    out["original_r2_bug"] = {
        "note": "MRzeroCore spin_dist='r2' builds spin_pos with torch.stack of "
                "three (S,) vectors -> shape (3, S) instead of (S, 3); the "
                "subsequent voxel_size broadcast then fails for all S != 3. "
                "The fast implementation uses stack(dim=1) and therefore "
                "implements the documented (S, 3) behaviour, so r2 runs "
                "there but crashes in the reference.",
        **r2_bug,
    }
    del data, seq

    # naive reference at very small scale
    if not args.quick or True:
        print("\n=== naive reference n=4 S=4 K=6 ===")
        n, S, K, seed = 4, 4, 6, 2
        data = build_phantom(n, seed=1000 + n)
        seq = to_mr0_seq(build_bssfp_seq(n, K, ti_start=30e-3), "bench_naive.seq")
        torch.manual_seed(seed)
        s_naive = naive_sim(seq, data, S, "rand")
        torch.manual_seed(seed)
        s_fast = isochromat_sim_fast(seq, data, S, print_progress=False,
                                     spin_dist="rand")
        # naive_sim derives spins from a fresh RNG draw with the same seed
        # and the same draw order; shapes must match
        assert s_naive.shape == s_fast.shape, (s_naive.shape, s_fast.shape)
        m = signal_metrics(s_fast, s_naive)
        m["case"] = {"n": n, "S": S, "K": K, "spin_dist": "rand"}
        out["naive_reference"] = m
        print(m)
        fig, ax = plt.subplots(figsize=(10, 4))
        npl = min(120, s_fast.shape[0])
        ax.plot(s_fast[:npl, 0].real.cpu(), label="fast (vectorized)", lw=1)
        ax.plot(s_naive[:npl, 0].real.cpu(), label="naive per-event reference",
                lw=1, alpha=0.8, ls="--")
        ax.set_xlabel("sample"); ax.set_ylabel("Re signal (coil 0)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORTS / "naive_reference.png", dpi=150)
        plt.close(fig)

    # ---------------- 3: physics check (IR null at T1 ln 2) ---------------
    if not args.quick:
        print("\n=== physics: IR null ===")
        t1 = 1.0
        tis = [f"{f * t1:.3f}" for f in (0.25, 0.40, 0.55, 0.6931, 0.80, 1.10)]
        data = build_phantom(32, seed=42, coil_count=1,
                             single_tissue=(t1, 0.1, 1.0))
        data.coil_sens = torch.ones_like(data.coil_sens)
        curve = []
        for ti in tis:
            seq = to_mr0_seq(build_bssfp_seq(32, 20, ti_start=float(ti)),
                             f"bench_ir_{ti}.seq")
            torch.manual_seed(0)
            sig = isochromat_sim_fast(seq, data, 64, print_progress=False)
            # first 5 readouts (20 samples each): median |signal|
            first = sig[: 5 * 32].reshape(-1, 32, 1).abs().mean(1)
            curve.append({"TI": float(ti),
                          "mag": float(first.median()),
                          "ti_over_t1": float(ti) / t1})
            print(curve[-1])
        out["ir_null"] = {"T1": t1, "curve": curve}
        best = min(curve, key=lambda c: c["mag"])
        out["ir_null"]["argmin_TI"] = best["TI"]
        out["ir_null"]["expected_TI"] = t1 * math.log(2)
        out["ir_null"]["null_depth_vs_flanks"] = {
            "null_mag": best["mag"],
            "flank_min": min(c["mag"] for c in curve
                             if c["TI"] not in (best["TI"],)),
        }
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([c["ti_over_t1"] for c in curve], [c["mag"] for c in curve],
                "o-")
        ax.axvline(math.log(2), color="r", ls="--",
                   label=f"T1·ln2 = {t1 * math.log(2):.3f} s")
        ax.set_xlabel("TI / T1")
        ax.set_ylabel("median |first-5 readouts|")
        ax.set_title("Inversion-recovery null (fast simulation only)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORTS / "ir_null_curve.png", dpi=150)
        plt.close(fig)

    # ---------------- 4: performance ---------------------------------------
    perf_cases = ([dict(n=8, K=60, S=64), dict(n=16, K=60, S=64)]
                  if args.quick else
                  [dict(n=8, K=60, S=64), dict(n=16, K=60, S=64),
                   dict(n=32, K=60, S=64), dict(n=64, K=100, S=64)])
    out["performance"] = []
    for case in perf_cases:
        print(f"\n=== perf n={case['n']} K={case['K']} S={case['S']} ===")
        data = build_phantom(case["n"], seed=1000 + case["n"])
        seq = to_mr0_seq(build_bssfp_seq(case["n"], case["K"]),
                         f"bench_perf_n{case['n']}.seq")
        st = seq_stats(seq)

        # warm up the GPU with the fast implementation (caches / clocks)
        torch.manual_seed(0)
        isochromat_sim_fast(seq, data, case["S"], print_progress=False)

        iters_ref = 1 if args.quick else (3 if case["n"] <= 32 else 1)
        torch.manual_seed(0)
        t_ref, mem_ref, _ = run_timed(
            lambda: mr0.isochromat_sim(seq, data, case["S"],
                                       print_progress=False),
            iters_ref, warmup=0)
        torch.manual_seed(0)
        t_fast, mem_fast, _ = run_timed(
            lambda: isochromat_sim_fast(seq, data, case["S"],
                                        print_progress=False),
            5, warmup=1)
        row = {"case": {k: case[k] for k in ("n", "K", "S")}, **st,
               "ref_time_s": t_ref, "fast_time_s": t_fast,
               "speedup": t_ref / t_fast,
               "ref_peak_mem_gib": mem_ref, "fast_peak_mem_gib": mem_fast,
               "ref_ms_per_event": 1000 * t_ref / st["n_events"],
               "fast_ms_per_event": 1000 * t_fast / st["n_events"]}
        out["performance"].append(row)
        print(row)
        del data, seq
        torch.cuda.empty_cache()

    # production-scale fast-only run at 128^3
    if not args.quick:
        print("\n=== fast-only production scale n=128 K=200 S=64 ===")
        data = build_phantom(128, seed=1000 + 128)
        seq = to_mr0_seq(build_bssfp_seq(128, 200), "bench_perf_n128.seq")
        st = seq_stats(seq)
        torch.manual_seed(0)
        t_fast, mem_fast, _ = run_timed(
            lambda: isochromat_sim_fast(seq, data, 64, print_progress=False),
            2, warmup=1)
        out["production_scale"] = {
            "case": {"n": 128, "K": 200, "S": 64}, **st,
            "fast_time_s": t_fast, "fast_peak_mem_gib": mem_fast,
            "fast_ms_per_event": 1000 * t_fast / st["n_events"]}
        print(out["production_scale"])
        # extrapolation for the full 128^3 / 6400-readout config:
        # per-event cost scales linearly with voxel count (V*S), so scale the
        # measured 128^3 per-event times by 6400*128 / n_events.
        n_events_target = 6400 * 128
        scale = n_events_target / st["n_events"]
        # reference per-event at 128^3 scaled up 8x from the measured 64^3
        perf64 = next(r for r in out["performance"] if r["case"]["n"] == 64)
        ref_ms_event_128 = perf64["ref_ms_per_event"] * 8
        fast_ms_event_128 = out["production_scale"]["fast_ms_per_event"]
        out["extrapolation_128_6400"] = {
            "note": "extrapolated: per-event cost scaled linearly in voxel count; "
                    "reference at 128^3 scaled 8x from measured 64^3 value",
            "ref_time_h": ref_ms_event_128 * n_events_target / 3.6e6,
            "fast_time_min": fast_ms_event_128 * n_events_target / 6e4,
        }
        print(out["extrapolation_128_6400"])
        del data, seq
        torch.cuda.empty_cache()

    # scaling figures
    rows = out["performance"]
    xs = [r["case"]["n"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].semilogx(xs, [r["ref_time_s"] for r in rows], "o-",
                   label="MRzeroCore reference")
    ax[0].semilogx(xs, [r["fast_time_s"] for r in rows], "s-",
                   label="isochromat_sim_fast")
    ax[0].set_xlabel("phantom size n (n^3 voxels)")
    ax[0].set_ylabel("simulation time [s]")
    ax[0].set_title("Wall time vs problem size")
    ax[0].legend()
    ax[1].semilogx(xs, [r["ref_ms_per_event"] for r in rows], "o-",
                   label="reference")
    ax[1].semilogx(xs, [r["fast_ms_per_event"] for r in rows], "s-",
                   label="fast")
    ax[1].set_xlabel("phantom size n (n^3 voxels)")
    ax[1].set_ylabel("ms per event")
    ax[1].set_title("Per-event cost")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "scaling_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    w = 0.38
    xx = np.arange(len(xs))
    ax.bar(xx - w / 2, [r["ref_peak_mem_gib"] for r in rows], w,
           label="reference", color="tab:blue")
    ax.bar(xx + w / 2, [r["fast_peak_mem_gib"] for r in rows], w,
           label="fast", color="tab:orange")
    ax.set_xticks(xx, [f"{n}³" for n in xs])
    ax.set_ylabel("peak memory delta [GiB]")
    ax.set_title("Peak GPU memory during simulation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "scaling_memory.png", dpi=150)
    plt.close(fig)

    path = REPORTS / "fast_iso_benchmark.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
