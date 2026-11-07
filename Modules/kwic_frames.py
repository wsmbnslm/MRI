import torch
import math
from typing import Optional, Tuple
import numpy as np
import Modules.bssfp_params as p
from Modules import PARAMS, build_sim_data

p=PARAMS
device = torch.device('cuda')


def build_kwic(mr_seq, p, device, N_center, N_max, frame_advance, DEPHASE=False):
    
    FIB = [8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    def _next_fib(x):
        return next(f for f in FIB if f >= x)

    k = torch.as_tensor(mr_seq.get_kspace()).float()
    k = k.reshape(-1, k.shape[-1])[:, :3].T.contiguous()
    k_full = k * (2 * np.pi * p.FOV / p.n)
    k_full = k_full.to(device)
    keep_full = k_full.norm(dim=0) <= np.pi
    k_norm = k_full / k_full.norm(dim=0).max() * 0.5     # cycles/voxel, for the CG operator
    rad = k_full.reshape(3, p.N_segments, p.N_readout, p.N_x)[:, 0, 0, :].norm(dim=0)
    r_frac = rad / rad.max()
    ring = np.array([_next_fib(max(N_center, int(np.ceil(N_max * float(r))))) for r in r_frac])
    
    djs, iis = [], []
    for jj in range(p.N_x):
        lo = -(int(ring[jj]) // 2)
        djs.extend(range(lo, lo + int(ring[jj])))
        iis.extend([jj] * int(ring[jj]))
    dj = np.asarray(djs, dtype=np.int64)     # (Kb,) echo-index offset
    ii = np.asarray(iis, dtype=np.int64)     # (Kb,) radial sample index
    Kb = dj.size
    
    # frame centers (echo index), windows clipped at the boundaries
    centers = list(range(0, p.N_readout, frame_advance))          # 0, 8, ..., 376
    n_frames = len(centers)
    ti_frames = np.array([p.TI_START + c * p.P_echo for c in centers])

    seg_off = (np.arange(p.N_segments)[:, None] * (p.N_readout * p.N_x))  # (S, 1)
    keep_np = keep_full.cpu().numpy()
    kwic_idx, kwic_w = [], []
    for c in centers:
        e = np.clip(c + dj, 0, p.N_readout - 1)                  # (Kb,) echo index
        flat = (seg_off + (e[None, :] * p.N_x + ii[None, :])).ravel()   # (S*Kb,)
        flat = flat[np.asarray(keep_np[flat], dtype=bool)]     # drop |k| > pi
        if DEPHASE:
            w = np.where(((flat // p.N_x) % p.N_readout) % 2 == 0, 1.0, -1.0).astype(np.float32)
        else:
            w = np.ones(len(flat), dtype=np.float32)
        kwic_idx.append(torch.as_tensor(flat, device=device))
        kwic_w.append(torch.as_tensor(w, device=device))
    K_frame = kwic_idx[0].numel()

    # k-space for the CG operator (cycles/voxel)
    kwic_k = [k_norm[:, idx] for idx in kwic_idx]

    return dict(k_full=k_full, keep_full=keep_full, k_norm=k_norm, kwic_idx=kwic_idx,    kwic_w=kwic_w, kwic_k=kwic_k, centers=centers, ti_frames=ti_frames, n_frames=n_frames, Kb=Kb)


    

