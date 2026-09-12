# Summary: Ehses et al., IR TrueFISP with a Golden-Ratio-Based Radial Readout (MRM 2013)

Source PDF: `Magnetic Resonance in Med - 2012 - Ehses - IR TrueFISP with a golden‐ratio‐based radial readout Fast quantification of T1 .pdf`

**Goal:** Simultaneously quantify relative proton density (M0), T1, and T2 in a **single ~6 s shot** per slice (20× faster than the previously required segmented Cartesian IR TrueFISP, ~2 min/slice), with values consistent with the literature.

## Core method

Inversion-recovery TrueFISP: an **inversion pulse** followed by a continuous series of **balanced SSFP (TrueFISP) readouts** with short, constant TR. The magnetization relaxes back toward steady state, and the readouts sample that transient. Parameters come from a 3-parameter mono-exponential fit to the reconstructed image time series.

### 1. Signal model (Eq. 1)

For on-resonant spins, signal after n TRs:

```
S(n·TR) = S_StSt − (S0 + S_StSt) · exp(−n·TR / T1*)
```

- `S_StSt`: steady-state signal
- `S0`: transient signal extrapolated to t = 0 (immediately after inversion)
- `T1*`: apparent relaxation time

### 2. Parameter extraction (Eqs. 2–4), assuming TR ≪ T2

With excitation flip angle α:

```
T1 = T1* · (S0 / S_StSt) · cos(α/2)
T2 = T1* · sin²(α/2) · (1 − (S_StSt / S0) · cos(α/2))⁻¹
M0 = S0 / sin(α/2)
```

- Flip-angle errors cause **T1↔T2 information leakage** → accurate (effective) flip angle is essential.
- T1 and T2 quantification are correlated.

### 3. Golden-ratio-based profile order

- Radial projections with **constant azimuthal spacing π/φ² ≈ 111.246°** (golden angle).
- Near-optimal uniform k-space coverage for **any** number of projections (most uniform at Fibonacci numbers), unlike interleaved equidistant ordering which is only optimal for one predetermined interleave count.
- Enables choosing the temporal frame rate **retrospectively** and per k-space location.

### 4. View-sharing: temporal k-space filter (modified k-space-weighted image contrast, Song & Dougherty)

- To build one time frame: k-space **center from ~8 projections** closest in time to the frame; **outer rings share views from farther away in time**.
- Ring sizes follow **Fibonacci numbers**: 8 (center), then jump to the next Fibonacci number (13, 21, 34, 55, …, up to 233) whenever the Nyquist criterion would otherwise be violated → each frame is (nearly) uniformly sampled.
- Filter is **shifted by 8 projections** per frame → temporal resolution **32–64 ms**.
- Result: many low-spatial-resolution, high-temporal-resolution images from one shot (e.g. 187 frames in the in vivo data).

### 5. Reconstruction

- **Nonuniform FFT gridding** (NUFFT, Fessler & Sutton toolbox).
- **Density compensation:** Ram-Lak (1/r) DCF **corrected for angular anisotropy** of the golden-ratio order — weighted by the angular distance to the two neighboring projections within each ring.
- Multichannel combine: **Walsh adaptive combine**.

### 6. Corrections (secondary, add scan time 6 → 18–28 s/slice)

**Slice profile imperfection** (nominal FA only reached at slice center):
- Define an **effective flip angle** α_eff(T1, T2) < α_nominal: the FA that best fits the slice-summed signal evolution assuming a perfect rectangular profile.
- Build a **lookup table** by simulating the IR TrueFISP transient (5× T1*) over a T1/T2 grid (T1 = 20–8000 ms, T2 = 5–2000 ms) using the measured slice profile.
- **Iterate:** initial T1/T2 from nominal FA → look up α_eff → refit → update; 4 iterations until mean α_eff change < 1%.

**Magnetization transfer (MT)** (MT lowers S_StSt and shortens T1*; S0 is largely unaffected):
- Acquire **3 separate IR TrueFISP experiments** with different RF pulse durations (stretch factor β = 1, 1.5, 2; TRF/TR kept fixed at 20%). Stretching the pulse by β reduces MT saturation rate per pulse by 1/β².
- Fit steady-state signals across β to the exponential model (Eq. 5):

```
S_StSt(β) = ΔS_StSt · (1 − e^(−k·β)) + S_StSt_fullMT
```

  → extrapolate to β → ∞ to get the **MT-free steady-state signal** S_StSt_noMT (3 parameters: ΔS, k, S_fullMT).
- Final maps use: **S_StSt_noMT** from the extrapolation, **T1\* from the scan with highest β** (least MT-affected), **S0 from the shortest-TR scan** (least blurring).
- Without correction: MT overestimates T1 by up to ~30% and underestimates T2 by up to ~50% in white matter.

## Sequence parameters used

**Phantom** (1.5 T, 6+ch body matrix coil; Resovist phantom, T1 316–2440 ms, T2 15–1680 ms):
- TR = 3.12 ms, FA = 45°, TRF = 0.8 ms, TBW = 2, 128×128, FOV 200×200 mm², slice 10 mm, BW 1000 Hz/px, **1800 projections, 5.7 s**.
- Reference: Cartesian spin echo with 5 inversion times (T1) / 5 echo times (T2).

**In vivo** (1.5 T, 32-ch head coil, 6 volunteers; with slice-profile + MT corrections):
- Adiabatic inversion, then **4-pulse linearly increasing flip-angle preparation** (suppress off-resonance oscillations).
- TR = 4 / 6 / 8 ms, TRF = 0.8 / 1.2 / 1.6 ms (TRF/TR = 20%), FA = 60°, TBW = 2, 256×256, FOV 220×220 mm², slice 6 mm, BW 890 Hz/px, total ~6 s per experiment.
- 187 reconstructed frames (TR = 4 ms scan), inversion times ranging 34.5 ms to 5.9 s.

## Key results

- Phantom T1/T2 agree with spin-echo reference after slice-profile correction (Table 1).
- In vivo WM/GM/CSF values consistent with published literature (Table 2).
- Whole-head coverage (20–30 slices) in 2–3 min → quantitative/synthetic imaging clinically viable.
- Without slice-profile/MT correction, quantification errors are considerable.

## Implementation notes (for a simple example)

Minimal pipeline (no MT / slice-profile corrections):
1. Simulate IR TrueFISP: inversion pulse + N bSSFP excitations with constant TR, golden-angle radial readout (pypulseq or direct k-space simulation).
2. Build golden-angle trajectory: projection i at angle `i * 111.246°` (mod 180°).
3. Temporal k-space filter per frame: 8 views for k-space center, Fibonacci-sized rings (13, 21, 34, …) out to max radius; advance by 8 views per frame.
4. Reconstruct each frame: NUFFT (e.g. `torchkbnufft`) + Ram-Lak DCF (with golden-ratio anisotropy correction for accuracy).
5. Pixel-wise 3-parameter fit of Eq. 1 → S_StSt, S0, T1\* → Eqs. 2–4 → M0, T1, T2 maps.
6. (Optional) slice-profile correction via α_eff lookup table; MT correction requires 3 scans with different β.

Open questions / details to check when implementing:
- Exact ring boundaries: radius where Nyquist would be violated given the current ring's angular span (Fig. 1b of the paper).
- First frames after inversion: filter is highly asymmetric (few views available before t=0) — how many initial frames are usable?
- DCF anisotropy correction: weighting by angular distance to the two neighboring projections per ring (Eq. in Fig. 1b).
- Coil sensitivity: paper uses Walsh adaptive combine; a simple example could use a single channel or a synthetic sensitivity map.
