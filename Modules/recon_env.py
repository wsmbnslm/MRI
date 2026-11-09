from pathlib import Path

import numpy as np
import torch

import MRzeroCore as mr0

import Modules.recon_ops as ro
from Modules import PARAMS, build_sim_data
from Modules.kwic_frames import build_kwic
from Modules.Plan import KbNufft, KbNufftAdjoint
from Modules.recon_ops import IdentityOperator, ChannelOperator, K3
from Modules.Regularizer import Regularizer


def setup_recon(repo_root, device, ns,
                seq_name="bssfp_kooshball3D_signal_n64_R384_S25.seq",
                signal_name="bssfp_kooshball3D_signal.pt",
                N_center=8, N_max=55, frame_advance=4,
                SEED=0, build_model=True, warm_start=True):
    """Build the full recon environment and publish it into ``ns``.

    Parameters
    ----------
    repo_root : str or Path
        Folder containing Modules/ (the repo root).
    device : torch.device
        Where everything except the phantom build lives (cuda).
    ns : dict
        Namespace to publish into - pass globals() from the notebook cell or
        script top level.

    Returns
    -------
    dict
        The same mapping that was published into ns.
    """
    repo_root = Path(repo_root)
    p = PARAMS

    # --- sequence + signal + phantom ---------------------------------------
    seq_file = repo_root / "Sequences" / "Sequence files" / seq_name
    if not seq_file.exists():
        raise FileNotFoundError(f"Missing .seq: {seq_file}\n"
                                f"Run 3D_bssfp_sequence.ipynb first.")
    mr_seq = mr0.Sequence.import_file(str(seq_file))

    signal_file = repo_root / "Simulation & Visualization" / "Signals" / signal_name
    if not signal_file.exists():
        raise FileNotFoundError(f"Missing .pt: {signal_file}\n"
                                f"Run Simulate.ipynb first.")
    signal = torch.load(signal_file)

    data, masks, brain_mask, coil_den, coil_c64 = build_sim_data(
        raw_path=str(repo_root / "Data" / "subject04_crisp_v.rawb"),
        p=p, device=torch.device("cpu"), permute_order=(2, 1, 0),
    )
    coil_den = coil_den.to(device)
    coil_c64 = coil_c64.to(device)

    # --- KWIC frames ---------------------------------------------------------
    kwic = build_kwic(mr_seq, p, device, N_center, N_max, frame_advance)
    kwic_idx, kwic_w, kwic_k = kwic["kwic_idx"], kwic["kwic_w"], kwic["kwic_k"]
    ti_frames, n_frames = kwic["ti_frames"], kwic["n_frames"]

    # --- protocol scalars the cells use by name -----------------------------
    n = p.n; FOV = p.FOV; N_x = p.N_x; N_readout = p.N_readout
    N_segments = p.N_segments; N_channels = p.N_channels; n_adc = p.n_adc
    TR_echo = p.TR_echo; TI_START = p.TI_START; P_echo = p.P_echo

    # --- per-frame flat data + shared NUFFT plans ---------------------------
    F_ALL = len(kwic_idx)
    dat, nKf = {}, {}
    for f in range(F_ALL):
        dat[f] = (signal[:, kwic_idx[f]] * kwic_w[f][None, :]).contiguous() \
                .view(torch.float32).ravel()
        nKf[f] = kwic_k[f].shape[1]
        assert nKf[f] == kwic_idx[f].numel(), f"frame {f}: stale kwic arrays"
    nufft3 = KbNufft(im_size=(n, n, n)).to(device)
    adj3 = KbNufftAdjoint(im_size=(n, n, n)).to(device)

    # --- operators -----------------------------------------------------------
    I_op = IdentityOperator((n, n, n))
    C_op = ChannelOperator(coil_c64, (n, n, n))
    # K3 / A_ops / get_partition read these as recon_ops module globals; publish
    # them BEFORE any operator call (the warm start below calls K3._rmatvec,
    # which looks up adj3 in the recon_ops namespace).
    ro.__dict__.update(dict(nufft3=nufft3, adj3=adj3, C_op=C_op, I_op=I_op,
                            kwic_idx=kwic_idx, kwic_w=kwic_w, kwic_k=kwic_k,
                            nKf=nKf, dat=dat, SEED=SEED))
    A_full = {f: K3(kwic_k[f]) @ C_op for f in range(F_ALL)}

    # --- regularizer (architecture train/inference must both match) ----------
    model = None
    if build_model:
        model = Regularizer((n, n, n, F_ALL), regularizer="ResNet", features=32,
                            num_of_resblocks=6, kernel_size=(3, 3, 3), scale_factor=1,
                            device=device, dtype=torch.complex64)

    # --- warm start: regrid at data scale ------------------------------------
    x_st = z_st = regrid = warm_scales = None
    if warm_start:
        x_st, z_st, regrid, warm_scales = [None] * F_ALL, [None] * F_ALL, {}, []
        for f in range(F_ALL):
            x0 = A_full[f].H @ dat[f]
            s = float(torch.linalg.norm(A_full[f] @ x0, 1) / torch.linalg.norm(dat[f], 1))
            x_st[f] = x0 / s
            z_st[f] = torch.zeros_like(x0)
            regrid[f] = torch.view_as_complex(x_st[f].reshape(-1, 2)).reshape(n, n, n)
            warm_scales.append(s)

    env = dict(mr_seq=mr_seq, signal=signal, data=data, masks=masks,
               brain_mask=brain_mask, coil_den=coil_den, coil_c64=coil_c64,
               kwic=kwic, kwic_idx=kwic_idx, kwic_w=kwic_w, kwic_k=kwic_k,
               ti_frames=ti_frames, n_frames=n_frames,
               n=n, FOV=FOV, N_x=N_x, N_readout=N_readout,
               N_segments=N_segments, N_channels=N_channels, n_adc=n_adc,
               TR_echo=TR_echo, TI_START=TI_START, P_echo=P_echo,
               F_ALL=F_ALL, dat=dat, nKf=nKf,
               nufft3=nufft3, adj3=adj3,
               I_op=I_op, C_op=C_op, A_full=A_full,
               model=model, x_st=x_st, z_st=z_st, regrid=regrid,
               warm_scales=warm_scales, SEED=SEED)
    ns.update(env)
    ro.__dict__.update(env)   # keep recon_ops globals in sync (incl. x_st etc.)
    return env
