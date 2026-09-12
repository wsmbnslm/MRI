# PDG loses the inversion-recovery transient in phase-cycled bSSFP (TrueFISP): null washed out, steady state ~50× too small

## Short summary

I am simulating the IR-TrueFISP sequence from Ehses et al. (MRM 69:71-81, 2013) with MRzeroCore 1.0.8. For the same sequence and phantom, the time-domain spin simulator (`isochromat_sim`) reproduces the expected signed, T1-differential inversion-recovery null, while the PDG (`compute_graph`/`execute_graph`) does not: it crosses zero within ~2 TR for all tissues and converges to a steady state that is 35-60× smaller than the spin sim. I could not find a practical parameter configuration that fixes this (see below). Is this expected/known behaviour?

## Environment

- MRzeroCore 1.0.8 (pip), pypulseq 1.5.0.post1, torch 2.14.0, Python 3.13.15, Linux (CPU)

## Setup

- **Sequence:** 180° inversion (10 ms sinc, TI = 20 ms) followed by 700 TrueFISP spokes (TR = 4 ms, FA = 60°, 64 samples/spoke, radial, 180° RF/ADC phase cycling every TR). On-resonant (B0 = 0), T2' dephasing disabled (T2dash = 1e3 s).
- **Phantom:** one voxel per tissue with the mean T1/T2/PD of the MNI 1.5 T brain phantom from `load_phantom`:
  WM 849 ms / 72.9 ms / 0.685, GM 1389 ms / 102.3 ms / 0.769, CSF 3032 ms / 388.6 ms / 0.873. B1 = 1, single coil.
- **Simulators compared:** `isochromat_sim(seq, data, spin_count=1, spin_dist="r2")` vs `compute_graph(seq, data)` + `execute_graph(graph, seq, data)` (defaults).
- A self-contained repro script is attached (`pdg_transient_repro.py`); the first two sections take ~5 min, the last one is documented below.

## Results (PDG defaults: max_state_count=200, min_state_mag=1e-4, gates 1e-2)

Per-spoke centre sample, real part. Null = first negative→positive zero crossing; time ≈ (spoke+1)·TR.

| tissue | sim | start | null | steady state |
|--------|-----|-------|------|--------------|
| WM  | spin sim | −1.66 | 368 ms  | **+0.243** |
| WM  | PDG      | −0.55 | 32 ms   | +0.007 |
| GM  | spin sim | −1.91 | 592 ms  | **+0.240** |
| GM  | PDG      | −0.64 | 24 ms   | +0.005 |
| CSF | spin sim | −2.23 | 1472 ms | **+0.253** |
| CSF | PDG      | −0.74 | 24 ms   | +0.004 |

1. The spin sim shows the expected physics: the signal starts inverted (negative), crosses zero at a T1-dependent time (WM < GM < CSF) and recovers to a positive steady state (paper Eq. 1 / Fig. 4b).
2. The PDG also starts negative, but crosses within ~2 TR for **all** tissues and settles at a steady state 35-60× smaller than the spin sim. The train runs ~2.8 s (≫ all T1s), so any correct state bookkeeping should converge to the same steady state — this is not only a transient issue.

## `max_state_count` does not converge (GM)

| msc   | 200   | 500   | 1000  | 1500  |
|-------|-------|-------|-------|-------|
| null  | 24 ms | 36 ms | 32 ms | 48 ms |
| SS    | +0.007 | +0.005 | +0.005 | +0.005 |

Larger state budgets do not move the result toward the spin-sim values. With the full 64×64 brain phantom the sweep is worse: a visible recovery at msc=150 collapses to *exactly zero* at msc=1500 (default gates); with a 7-band vial phantom the steady state decreases monotonically with msc (3.53 → 3.38 → 2.43 for msc 150 → 300 → 750).

## The documented "accurate" settings are impractical for this sequence

Per the *State selection* docs (`max_state_count=5000, min_state_mag=1e-12`, `min_emitted_signal=1e-3, min_latent_signal=1e-3`): the Rust pre-pass takes ~50 s, but the `execute_graph` main pass then runs at **~1 repetition/s** for this 721-repetition sequence (≈ 2 h per tissue, ~6 h for the three tissues) — vs ~30 s for the spin sim. It appears the per-state loop becomes the bottleneck as the state pool grows.

## Mechanism (from reading `src/pre_pass.rs`)

My best guesses for where the transient is lost:

- The pre-pass, after **every** RF pulse, sorts the states by |mag|² (f32) and truncates to the top `max_state_count` — a greedy, myopic pruning that can drop states that matter later or for coherent summation.
- The magnitude estimates used for pruning are computed with the **mean** T1/T2 of the phantom, so one graph (pruned for average tissue) is applied to all tissues.
- The main pass gates states by *relative* emitted/latent signal per repetition (default 1e-2). As the state pool grows, more states fall under the relative threshold — this may explain the collapse-to-zero when msc is increased while keeping the default gates.
- Coherent sums are accumulated in f32.

## Questions

1. Is loss of the transient for a long, phase-cycled bSSFP train with an inversion pulse expected/known for the PDG?
2. Is there a configuration under which the PDG agrees with the spin sim for this case, or should transient simulation be considered out of scope (I would then use `isochromat_sim` for this use case)?
3. If this is a bug, is the msc non-convergence / collapse-to-zero a useful additional data point?

Happy to share the full script, outputs, or a reduced minimal sequence if that would help. Thanks!
