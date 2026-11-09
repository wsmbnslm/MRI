# End-to-end pipeline for sequence generation, simulation, KWIC frame reconstruction, model training & inference:

```
sequence design -> phantom simulation -> KWIC frame assembly
-> zero-shot self-supervised DeepADMM training -> inference / plotting
```

## Repository structure

```
Sequences/
    Sequence generation notebooks        protocol design (MRzeroCore) -> .seq file
    Sequence files/
        generated .seq files

Simulation & Visualization/
    Simulation notebook                  Bloch-equation simulation of the .seq on the
                                         phantom -> multi-coil signal
    Signals/
        generated .pt files

KWIC/
    Notebook to build KWIC frames        interactive KWIC (k-space windowing in
                                         time) frame assembly + recon previews

Training & Inference/
    Model training notebook              notebook port of train.py (4 cells:
                                         imports -> setup_recon -> train setup -> loop)
    Python scripts to run in tmux        same training as a script (tmux-friendly);
                                         the reference implementation
    Trained models/                      trained models here with logs

Results & Metrics/
    Notebooks to plot model's
    reconstruction                       loads a checkpoint (or the final-sweep),
                                         runs the inference sweep,
                                         plots ZS-SSL vs regrid mid-slices

Data/
    subject04_crisp_v.rawb               brain phantom (PD/T1/T2/B0 + masks)

Papers/                                  the three reference papers

Benchmark & Reports/                     NUFFT fast-iso benchmark (standalone)

Modules/                                 all shared code (imported by every notebook)
    bssfp_params.py                      BSSFPParams protocol object (PARAMS) - the
                                         single source of truth for n, FOV, N_channels,
                                         N_readout, N_segments, TR, TI, ...
    recon_env.py                         setup_recon(): ONE call builds the whole
                                         environment (sequence, signal, phantom, KWIC,
                                         flat data, NUFFT plans, operators, ResNet,
                                         warm-start regrid) for training AND inference
    phantom.py                           build_sim_data(): phantom + coil sensitivities
    kwic_frames.py                       build_kwic(): KWIC butterfly geometry
    recon_ops.py                         LinearOperator algebra (@, .H, +, *),
                                         ChannelOperator (SENSE), K3 (coil-batched 3D
                                         NUFFT), A_ops, get_partition, _sub
    Plan.py                              KbNufft / KbNufftAdjoint (Kaiser-Bessel
                                         table NFFT plans)
    FourierTransform.py                  nonuniform_fourier_transform_forward/_adjoint
    NumSolvers.py                        conjugate_gradient
    Regularizer.py                       Regularizer wrapper (ResNet or UNet over the
                                         full frame stack, complex64)
    ResNet.py / UNet.py                  the networks
    DeepADMM.py                          _admm_iter: one dual-free ADMM iteration
                                         (CG x-update + net z-update with temporal mask)
    system.py                            MRzeroCore system defaults
    CoilSens.py, ModelconvLayer.py,      helpers / legacy pieces
    fast_iso.py
```

## Tutorial: running the pipeline

**Environment:** the conda env with torch (CUDA), torchkbnufft, MRzeroCore,
pypulseq, scipy. Everything runs on one GPU; a training run uses ~1-2 GB
VRAM. Notebooks and scripts bootstrap the repo path themselves
(`_find_repo_root`), so run from any folder **INSIDE** the repo tree.

### STEP 1 - sequence (once)

```
Sequences/3D_bssfp_sequence.ipynb
```

→ writes `Sequences/Sequence files/bssfp_kooshball3D_signal_n64_R384_S25.seq`

### STEP 2 - simulate (once per protocol change)

```
Simulation & Visualization/Simulate.ipynb
```

→ writes `Simulation & Visualization/Signals/bssfp_kooshball3D_signal.pt`

### STEP 3 - KWIC frames (exploratory)

```
KWIC/Fibonacci KWIC.ipynb
```

Assembles the 96 KWIC frames (`N_center=8`, `N_max=55`, `frame_advance=4`)
and shows recon previews. Not required for training (training rebuilds the
frames itself via `Modules/kwic_frames.build_kwic`).

### STEP 4 - train

Either run `Training & Inference/Model_training.ipynb` top to bottom, or
(recommended for long runs) the script in a tmux session:

```bash
tmux new -s train
cd <repo>/Training\ \&\ Inference
python train.py 2>&1 | tee train2.log
```

Useful knobs at the top of `train.py`:

```python
N_STEPS, K_PART, LR, RHO, SEED     # hyperparameters
SMOKE = True                       # 5-step sanity run
RESUME = "zs3d_ckpt_75.pt"         # resume a killed/finished run
ELBOW = True/False                 # L-curve auto-stop (progress-gated)
```

Checkpoints every 25 steps: `zs3d_ckpt_<step>.pt` in the CWD; the final
sweep writes `zs3d_frames.pt`.

### STEP 5 - inference / plots

```
Results & Metrics/Plot_inference.ipynb
```

Runs `setup_recon()`, loads a checkpoint (`zs3d_ckpt_*.pt`) or the saved
final sweep (`zs3d_frames.pt`), does the full-k-space ADMM sweep, and
plots ZS-SSL vs regrid mid-slices across the TI span.

## Conventions & gotchas

- `setup_recon(REPO_ROOT, device, ns=globals())` is the single environment
  builder for BOTH training and inference notebooks; it also publishes the
  variables into the `recon_ops` module namespace (the operator functions
  read them as globals). Rebuild operator instances after re-running imports.
- Folder names matter exactly as written (some contain spaces:
  `Simulation & Visualization`, `Training & Inference`, `Results & Metrics`,
  `Sequence files`, `Trained models`, `Benchmark & Reports`).
- `train.py` / notebooks write outputs to the **CURRENT WORKING DIRECTORY**,
  not to the folder the script lives in — launch from wherever you want the
  checkpoints.
- Checkpoint files are large (~415 MB each: full ADMM state + history);
  `zs3d_frames.pt` is ~200 MB.
- Long runs: use tmux + `tee` for the log; `RESUME` makes runs crash-proof.
- Jupyter text-editor autosave has truncated files on this NFS before —
  keep the repo committed (or at least checkpointed) for anything valuable.

## TMUX Cheat Sheet

| action | command |
|---|---|
| start a new session | `tmux new -s mysession` |
| kill a session | `tmux kill-session -t mysession` |
| view session list | `tmux ls` |
| attach to a session | `tmux attach-session` |
| deattach from a session | `Ctrl + b` then `d` |
