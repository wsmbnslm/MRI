import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchkbnufft as tkbn
import pypulseq as pp
import MRzeroCore as mr0
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for d in [cur, *cur.parents]:
        if (d / "Modules").is_dir():
            return d
    raise RuntimeError(f"Could not find a folder containing 'Modules' from {cur}") 

REPO_ROOT = _find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Drop any cached Modules* modules so a kernel restart is only needed for
# *new* modules, not for edits to existing ones - re-running this cell
# always re-imports the current Modules/ code.
for _m in [m for m in list(sys.modules) if m == 'Modules' or m.startswith('Modules.')]:
    del sys.modules[_m]

from Modules import PARAMS, build_sim_data
from Modules.bssfp_params import TISSUES, T2STAR, NAMES
from Modules.recon_ops import LinearOperator, ChannelOperator, IdentityOperator, recon_nufft, recon_cg, signed_series, mag_series, get_partition, K3, A_ops, _sub
from Modules.Plan import KbNufft, KbNufftAdjoint
from Modules.NumSolvers import conjugate_gradient
from Modules.kwic_frames import build_kwic
from Modules.Regularizer import Regularizer 
from Modules.DeepADMM import _admm_iter

torch.set_default_device("cuda")
device = torch.device("cuda")

seq_name = "bssfp_kooshball3D_signal_n64_R384_S25.seq"   # <-- choose sequence
seq_file = REPO_ROOT / "Sequences" / "Sequence files" / seq_name
mr_seq = mr0.Sequence.import_file(str(seq_file))   # <-- import the sequence
signal_name = "bssfp_kooshball3D_signal.pt"   # <-- choose signal
signal_file = REPO_ROOT / "Simulation & Visualization" / "Signals" / signal_name
signal = torch.load(signal_file)
data, masks, brain_mask, coil_den, coil_c64 = build_sim_data(
    raw_path=str(REPO_ROOT / "Data" / "subject04_crisp_v.rawb"),  # <-- <repo>/Data/
    p=PARAMS,
    device=torch.device("cpu"),
    permute_order=(2, 1, 0),
)

coil_den = coil_den.to(device)
coil_c64 = coil_c64.to(device)
# Bind the grid / readout names the cells below already use (same values as the
# sequence notebook, straight from the shared protocol params):
n          = PARAMS.n
FOV        = PARAMS.FOV
N_x        = PARAMS.N_x
N_readout  = PARAMS.N_readout
N_segments = PARAMS.N_segments
N_channels = PARAMS.N_channels
n_adc      = PARAMS.n_adc
TR_echo    = PARAMS.TR_echo
TI_START   = PARAMS.TI_START
P_echo     = PARAMS.P_echo

N_center = 8      # echoes for the k-space center
N_max    = 55     # outermost Fibonacci ring (= time window at the periphery)
frame_advance = 4 # echo-index advance per frame (8 * 12.65 ms = ~101 ms)
kwic = build_kwic(mr_seq, PARAMS, device, N_center, N_max, frame_advance)
kwic_idx, kwic_w, kwic_k = kwic["kwic_idx"], kwic["kwic_w"], kwic["kwic_k"]
ti_frames, n_frames = kwic["ti_frames"], kwic["n_frames"]

import Modules.recon_ops as ro
ro.__dict__.update({"kwic_idx": kwic_idx, "kwic_w": kwic_w, "kwic_k": kwic_k})

p = PARAMS
torch.manual_seed(0)
np.random.seed(0)
F_ALL = len(kwic_idx)
dat, nKf = {}, {}
for f in range(F_ALL):
    dat[f] = (signal[:, kwic_idx[f]] * kwic_w[f][None, :]) \
                    .contiguous().view(torch.float32).ravel()
    nKf[f] = kwic_k[f].shape[1]
    assert nKf[f] == kwic_idx[f].numel(), f"frame {f}: stale kwic arrays; re-run the frames cell"
print(f"[3D] K/frame (per coil): min={min(nKf.values())} max={max(nKf.values())}")

# subset-k NUFFT with ONE shared plan pair (operator objects are cheap)
nufft3 = KbNufft(im_size=(p.n, p.n, p.n)).to(device)
adj3 = KbNufftAdjoint(im_size=(p.n, p.n, p.n)).to(device)

B_FRAMES, U_ITERS, CG_ITERS = 8, 15, 2
N_STEPS, K_PART, LR, RHO, SEED = 300, 10, 5e-4, 1e-5, 0
I_op = IdentityOperator((p.n, p.n, p.n))
C_op = ChannelOperator(coil_c64, (p.n, p.n, p.n))

import Modules.recon_ops as ro
ro.__dict__.update(globals())

_Q, _At, _Al, _Av = A_ops(0, 0)
_x0 = torch.zeros(2 * n**3, dtype=torch.float32, device=device)
assert (_Q @ _x0).numel() == 2 * n**3, "Q@x size mismatch"
print(f"[3D] A_ops(0,0) ok: Q@x={(_Q @ _x0).numel()} (=2 n^3), "
      f"A_tr@x={(_At @ _x0).numel()}, A_lo@x={(_Al @ _x0).numel()}")
del _Q, _At, _Al, _Av, _x0

model = Regularizer((n, n, n, F_ALL), regularizer="ResNet", features=32,
                    num_of_resblocks=6, kernel_size=(3, 3, 3), scale_factor=1,
                    device=device, dtype=torch.complex64)
model.regularizer.layer1.convlayer.weight.data.zero_()
print(f"[3D] ResNet F={F_ALL}: {sum(pp.numel() for pp in model.parameters()):,} params (layer1 zeroed)")

A_full = {f: K3(kwic_k[f]) @ C_op for f in range(F_ALL)}
x_st, z_st, sc = [None] * F_ALL, [None] * F_ALL, []
t1 = time.time()
for f in range(F_ALL):                               # warm start: regrid, at data scale
    x0 = A_full[f].H @ dat[f]
    s = float(torch.linalg.norm(A_full[f] @ x0, 1) / torch.linalg.norm(dat[f], 1))
    x_st[f] = x0 / s; z_st[f] = torch.zeros_like(x0); sc.append(s)
regrid = {f: torch.view_as_complex(x_st[f].reshape(-1, 2)).reshape(n, n, n) for f in range(F_ALL)}
print(f"[3D] warm regrid scale ||Ax0||/||d|| median={np.median(sc):.3g} (divided out) ({time.time()-t1:.0f} s)")
print(f"[3D] GPU {torch.cuda.memory_allocated()/2**30:.2f} GB")

SMOKE = False            # 5 steps -> then set False for the full run
RESUME = "zs3d_ckpt_75.pt"           # resume the 75-step run; set None for a fresh run (SMOKE must be False)
model.train()
opt = torch.optim.Adam(model.parameters(), LR)
params = list(model.parameters())
history, t0, best_val, best_step, start = [], time.time(), float("inf"), -1, 0
if RESUME and not SMOKE:
    _st = torch.load(RESUME, map_location=device, weights_only=False)
    model.load_state_dict(_st["model"]); opt.load_state_dict(_st["opt"])
    x_st, z_st = _st["x_st"], _st["z_st"]
    history, best_val, best_step, start = _st["history"], _st["best_val"], _st["best_step"], _st["step"]
    print(f"[3D] resumed from {RESUME} at step {start}")

B_FRAMES, U_ITERS, CG_ITERS = 8, 15, 2
N_STEPS, K_PART, LR, RHO, SEED = 300, 10, 5e-4, 1e-5, 0

import Modules.DeepADMM as da
da.__dict__.update(globals())

ELBOW = False          # auto-stop off: on the old rotating-batch val it fired on noise
VAL_FRAMES, VAL_K = list(range(48, 56)), 3   # FIXED val set -> low-variance L-curve
val_ops, val_data = [], []
for f in VAL_FRAMES:
    _tr, _va, _lo = get_partition(f, VAL_K)
    val_ops.append(A_ops(f, VAL_K)[3])        # Av on the fixed partition
    val_data.append(_sub(dat[f], _va, nKf[f]))

N_run = 5 if SMOKE else N_STEPS
for step in range(start, N_run):
    k = step % K_PART
    rng_s = np.random.default_rng(SEED * 100003 + step)
    batch = [int(v) for v in rng_s.choice(F_ALL, B_FRAMES, replace=False)]
    h = batch[step % B_FRAMES]
    bidx = {f: i for i, f in enumerate(batch)}
    ops = [A_ops(f, k) for f in batch]                       # per-frame (per-frame k)
    b_f, dlo = [], []
    for i, f in enumerate(batch):
        tr, va, lo = get_partition(f, k)
        inv = 1.0 / (N_channels * nKf[f])
        b_f.append((ops[i][1].H @ _sub(dat[f], tr, nKf[f])) * inv)
        dlo.append(_sub(dat[f], lo, nKf[f]))

    opt.zero_grad()
    xs, zs = [x_st[f] for f in batch], [z_st[f] for f in batch]
    for _ in range(U_ITERS):
        xs, zs = checkpoint(_admm_iter, xs, zs, ops, batch, h, bidx, b_f,
                            use_reentrant=False)
    loss = sum(torch.linalg.norm(ops[i][2] @ xs[i] - dlo[i], 1) / torch.linalg.norm(dlo[i], 1)
               for i in range(len(batch))) / len(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 10.0)
    opt.step()
    with torch.no_grad():
        zr = max(float(zs[i].norm() / xs[i].norm().clamp(min=1e-12)) for i in range(len(batch)))
    for i, f in enumerate(batch):
        x_st[f], z_st[f] = xs[i].detach(), zs[i].detach()
    with torch.no_grad():                              # FIXED val set (low-variance L-curve)
        val = sum(float(torch.linalg.norm(val_ops[i] @ x_st[VAL_FRAMES[i]] - val_data[i], 1) /
                 torch.linalg.norm(val_data[i], 1)) for i in range(len(VAL_FRAMES))) / len(VAL_FRAMES)
    history.append((step, float(loss), float(val), h))
    print(f"[3D] step {step:3d} k={k} h={h:2d} loss={float(loss):.4f} val={float(val):.4f} "
          f"z/x={zr:5.2f} ({(time.time()-t0)/60:.1f} min)")
    if SMOKE:
        continue
    if ELBOW and len(history) >= 5 and step >= 50:      # L-curve elbow (same rule as 2D)
        roll = float(np.mean([r[2] for r in history[-5:]]))
        if roll < best_val: best_val, best_step = roll, step - 4
        # progress gate: only allow stopping if the curve actually fell vs its
        # warmup level (prevents firing on flat noise)
        if (best_val < 0.85 * float(history[4][2]) and roll > 1.03 * best_val
                and step - best_step > 25):
            print(f"[3D] elbow at ~{best_step} -> stopping"); break
    if (step + 1) % 25 == 0:
        torch.save({"step": step + 1, "model": model.state_dict(), "opt": opt.state_dict(),
                    "x_st": x_st, "z_st": z_st, "history": history,
                    "best_val": best_val, "best_step": best_step},
                   f"zs3d_ckpt_{step+1}.pt")
        s_, l_, v_, _h = zip(*history)
        plt.figure(figsize=(8, 3))
        plt.plot(s_, l_, lw=.7, alpha=.5, label="loss"); plt.plot(s_, v_, lw=.7, alpha=.5, label="val")
        plt.xlabel("step"); plt.legend(); plt.grid(alpha=.3); plt.title("3D ZS-SSL L-curve"); plt.show()

if not SMOKE:
    with torch.no_grad():                               # final sweep: full k-space, no mask
        for _ in range(U_ITERS):
            for f in range(F_ALL):
                A_f = A_full[f]; inv = 1.0 / (N_channels * nKf[f])
                x_st[f] = conjugate_gradient((A_f.H @ A_f) * inv + RHO * I_op,
                                             (A_f.H @ dat[f]) * inv + RHO * z_st[f],
                                             x_st[f], 10, True)
            zin = torch.stack([torch.view_as_complex(x_st[f].reshape(-1, 2)).reshape(n, n, n)
                               for f in range(F_ALL)])
            y = model(torch.view_as_real(zin.permute(1, 2, 3, 0).reshape(-1)).reshape(-1))
            zall = torch.view_as_complex(y.reshape(-1, 2)).reshape(n, n, n, F_ALL).permute(3, 0, 1, 2)
            z_st = [torch.view_as_real(zall[f].reshape(-1)).reshape(-1) for f in range(F_ALL)]
    zs3d = {f: torch.view_as_complex(x_st[f].reshape(-1, 2)).reshape(n, n, n) for f in range(F_ALL)}
    torch.save(zs3d, "zs3d_frames.pt")