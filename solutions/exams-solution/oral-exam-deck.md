# Structured Oral Exam — Magnetresonanztomographie 2

**Format:** At the start of preparation time, the student draws only the Category I card, which names a Jupyter notebook to complete during the 30-minute preparation window (see that category's intro). The Category II and Category III cards are drawn later, at the oral exam table itself. Each card has 3 questions, vertically drilling deeper into one topic. Category II and III cards are worth 4 + 3 + 3 = 10 points each. Category I cards are worth substantially more (14 + 3 + 3 = 20 points), since their first question also covers evaluating the 30-minute practical task itself, discussed and judged holistically rather than graded separately. Total: 40 points.

**Joker:** For the Category II and Category III cards only, the student may redraw one card, once, at the exam table. The Category I card is fixed once drawn at the start of preparation time and cannot be redrawn, since redrawing would mean switching notebooks with no time left to redo the practical task.

**Scoring philosophy:** Award partial credit within each question's point budget based on how many of the listed anticipated points the student volunteers spontaneously. Where a point is marked "if not mentioned, probe," the examiner only asks that follow-up if the student's own answer hasn't already addressed it — don't read it out by default, since it can tip off the answer. Each card is self-contained and does not require knowledge of what another card covers.

---

## CATEGORY I — Sequence Assembly

**How this category works:** at the start of the exam, the student draws only this category's card and goes to the preparation room. The card names a specific Jupyter notebook — the student completes the sequence-assembly task in that notebook during the 30-minute preparation window, and brings the finished notebook (and its resulting image) into the exam. The Category II and Category III cards are drawn later, at the oral exam table, and are answered close to cold. This category's card is not eligible for the joker redraw.

---

### Card I-A: Cartesian GRE — Notebook `mock-exam-1` (Cartesian GRE Sequence Assembly)

**Preparation:** Complete the sequence-assembly task in `mock-exam-1`. Bring the completed notebook and the resulting image.

**Q1 (14 pts).** Show your reconstructed image. Then walk through the loop you wrote: what determines the phase-encoding area in each iteration, and why are `gx_pre` and `gy_pre` added together in the same `seq.add_block()` call, rather than one after another?

*Anticipated points:*

- The notebook runs and produces a correctly reconstructed, recognizable image — judged holistically through discussion with the student rather than a strict pass/fail: full marks for a clean, fully correct solution; partial credit for an implementation with the right structure but a minor bug, or one where the student can correctly diagnose what went wrong even if not fully fixed (8 pts)
- Explains that `phase_areas` steps symmetrically from $-N_y/2$ to $N_y/2$ (scaled by $\Delta k_y$), so each loop iteration encodes a different $k_y$ line (2 pts)
- Explains that `gx_pre` and `gy_pre` are meant to occupy the same time slot rather than being played sequentially — playing them one after another would double the prephasing duration and push the echo later than the `te_delay` calculation assumes, so they need to be passed to the same `add_block()` call to stay consistent with that calculation (2 pts)
- Correctly relates `gy_pre`'s per-iteration area back to their own `phase_areas` computation (2 pts)

**Q2 (3 pts).** This notebook uses $T_\mathrm{R}=5\,\mathrm{s}$ and $T_\mathrm{E}=5\,\mathrm{ms}$. Given these specific values, what kind of image weighting would you expect, and why?

*Anticipated points:*

- $T_\mathrm{R}=5\,\mathrm{s}$ is far longer than typical tissue $T_1$ values, so longitudinal magnetization recovers nearly fully between repetitions regardless of tissue — minimal $T_1$-weighting (1 pt)
- $T_\mathrm{E}=5\,\mathrm{ms}$ is short, so little $T_2^*$ decay occurs before readout — minimal $T_2^*$-weighting (1 pt)
- Concludes this combination produces predominantly proton-density-weighted contrast (1 pt)

**Q3 (3 pts).** Given how long $T_\mathrm{R}$ is in this notebook (5 s), is the `gx_spoiler` gradient actually doing meaningful work here? Explain what a spoiler is generally needed for, and why this notebook's specific $T_\mathrm{R}$ makes that concern less pressing.

*Anticipated points:*

- Explains the spoiler's general purpose: dephasing/destroying residual transverse magnetization at the end of each TR, so it doesn't contaminate the next excitation (1 pt)
- Correctly reasons that residual transverse magnetization decays via $T_2$/$T_2^*$ relaxation on the order of tens to hundreds of milliseconds for most tissue — far shorter than this notebook's 5 s $T_\mathrm{R}$ (1 pt)
- Concludes that with $T_\mathrm{R}$ this long, transverse magnetization has already decayed away naturally well before the next repetition, so the spoiler is largely redundant here — even though it's essential in the short-TR spoiled GRE sequences this design is normally used for (1 pt)

---

### Card I-B: EPI — Notebook `mock-exam-2` (EPI Sequence Assembly)

**Preparation:** Complete both parts of the sequence-assembly task in `mock-exam-2`. Bring the completed notebook and the resulting image.

**Q1 (14 pts).** Show your reconstructed image. Then walk through your solution: why does the assembly loop only run `n_y // (2*R)` times instead of `n_y` times, and how did you derive the area of `gy_blip`?

*Anticipated points:*

- The notebook runs and produces a correctly reconstructed, recognizable image — judged holistically through discussion with the student rather than a strict pass/fail: full marks for a clean, fully correct solution; partial credit for an implementation with the right structure but a minor bug, or one where the student can correctly diagnose what went wrong even if not fully fixed (8 pts)
- Explains that each loop iteration acquires two lines (one forward via `gx`, one reversed via `gx_rev`), so only half as many iterations are needed to cover all `n_y` lines (2 pts)
- Explains that the readout gradient itself alternates direction between lines, rather than returning to its start each time, which is why `gx`/`gx_rev` alternate instead of always using the same gradient (2 pts)
- Correctly relates their own `gy_blip` area ($R\cdot\Delta k_y$) to stepping $k_y$ by exactly one line per blip (2 pts)

**Q2 (3 pts).** The reconstruction flips every second line (`kspace[1::2, :] = kspace[1::2, ::-1]`) before applying the FFT. Why is this necessary, and what would the reconstructed image look like if this step were skipped?

*Anticipated points:*

- Explains that every second line was acquired with the readout gradient reversed, so its samples run from $+k_{x,\text{max}}$ to $-k_{x,\text{max}}$ instead of the other direction — the raw sample order alternates between lines (1 pt)
- Explains that flipping restores a consistent $k_x$ ordering across all lines, which the FFT requires (it assumes a uniformly-ordered k-space grid) (1 pt)
- Explains the consequence of skipping this step: treating alternating lines as if they weren't reversed is equivalent to introducing a sign that flips with every line across $k_y$, which corresponds to a shift of half the FOV in image space — producing a duplicated, overlapping "ghost" image rather than merely a blurrier or shifted single image (1 pt)

**Q3 (3 pts).** This notebook computes `te_min` from your `gy_blip` and the number of lines. What does `te_min` represent physically, and why does a longer echo train make EPI more sensitive to $B_0$ inhomogeneity?

*Anticipated points:*

- `te_min` represents the earliest possible time from excitation to the center of the readout train (i.e. when $k_y=0$ is sampled), since that is what determines the achievable effective echo time (1 pt)
- A longer echo train (more lines, or slower blips) pushes this center-of-k-space timing later, increasing the minimum achievable $T_\mathrm{E}$ (1 pt)
- Explains that because there is no refocusing pulse repeated across this whole duration, phase errors from $B_0$ inhomogeneity accumulate progressively over the echo train, causing geometric distortion and/or signal dropout (1 pt)

---

### Card I-C: Radial GRE — Notebook `mock-exam-3` (Radial GRE Sequence Assembly)

**Preparation:** Complete the sequence-assembly task in `mock-exam-3`. Bring the completed notebook and the resulting image.

**Q1 (14 pts).** Show your reconstructed image. Then walk through your solution: why do the spoke angles only range from $0$ to $\pi$ rather than $0$ to $2\pi$, and why must `gx_pre`/`gy_pre` and `gx_ro`/`gy_ro` be rebuilt inside every loop iteration rather than reused from a single fixed gradient?

*Anticipated points:*

- The notebook runs and produces a correctly reconstructed, recognizable image — judged holistically through discussion with the student rather than a strict pass/fail: full marks for a clean, fully correct solution; partial credit for an implementation with the right structure but a minor bug, or one where the student can correctly diagnose what went wrong even if not fully fixed (8 pts)
- Explains that a spoke at angle $\theta$ and one at $\theta+\pi$ traverse the same line through k-space (just in opposite direction), so angles beyond $\pi$ would be redundant (2 pts)
- Explains that the projected area ($\cos\theta$, $\sin\theta$ scaling) is different for every spoke, so a new gradient waveform reflecting that specific angle must be built each iteration — a single fixed gradient object could not represent a rotating spoke (2 pts)
- Correctly relates their own `gx_pre`/`gy_pre`/`gx_ro`/`gy_ro` construction to this per-angle projection (2 pts)

**Q2 (3 pts).** Every spoke in this sequence passes through the center of k-space. What practical benefit does this redundancy provide, particularly with respect to patient motion?

*Anticipated points:*

- Radial sampling naturally oversamples the center of k-space, since every spoke passes through it (1 pt)
- This makes it more robust to motion: data corrupted by motion during a single spoke contributes incoherently/diffusely across the image, rather than causing a single coherent shift artifact (1 pt)
- Contrasts with Cartesian sampling, where motion during acquisition of one phase-encoding line primarily produces coherent artifacts along the phase-encoding direction (1 pt)

**Q3 (3 pts).** The reconstruction uses a NUFFT adjoint rather than the standard `np.fft.ifft2` used in the Cartesian notebook. Why can't the standard FFT approach be applied here directly, and what would happen if you tried?

*Anticipated points:*

- Explains that radial samples don't lie on a regular Cartesian grid, whereas the standard FFT algorithm assumes uniformly-spaced samples (1 pt)
- Explains that treating radial samples as if they were on a Cartesian grid would assign the wrong spatial frequency to each sample, producing a severely distorted or incorrect image — not merely a blurrier version of the correct one (1 pt)
- Correctly connects this to why a NUFFT (or gridding onto a Cartesian grid first) is required to properly account for the true, non-uniform sample locations (1 pt)

---

## CATEGORY II — Data Fidelity & Parallel Imaging

**How this category works:** drawn at the exam table, answered close to cold. The joker redraw applies to this category: one redraw, once, across Category II and III combined.

---

### Card II-A: The Linear Forward Model & Adjoints

**Q1 (4 pts).** For parallel-imaging MRI, the measured k-space data $\mathbf{d}$ is related to the image $\mathbf{x}$ through three operators: the channel operator $\mathbf{C}$, the Fourier operator $\mathbf{F}$, and the sampling operator $\mathbf{M}$. Write down the linear forward model using these three operators, explain what each one does, and describe the order in which they are applied.

*Anticipated points:*

- $\mathbf{C}$ (channel operator) is applied first: multiplies the image $\mathbf{x}$ by each coil sensitivity map, producing coil-weighted images (1 pt)
- $\mathbf{F}$ (Fourier operator) is applied next, transforming each coil image into its full k-space (1 pt)
- $\mathbf{M}$ (sampling operator) is applied last: an elementwise multiplication by a binary mask of 1s (acquired) and 0s (not acquired), zeroing out all unacquired k-space entries while keeping the array the same size (1 pt)
- If not mentioned, probe: "Could $\mathbf{M}$ and $\mathbf{F}$ be swapped, since both operate in k-space?" — correct answer: no, $\mathbf{F}$ must produce the full k-space before $\mathbf{M}$ can select the acquired subset; the order matters (1 pt)

**Q2 (3 pts).** What are the adjoints of $\mathbf{F}$, $\mathbf{M}$, and $\mathbf{C}$, and is the adjoint of an operator generally the same as its inverse?

*Anticipated points:*

- $\mathbf{F}^\mathrm{T}$ is the inverse Fourier transform (since $\mathbf{F}$ is unitary, adjoint = inverse here specifically) (1 pt)
- $\mathbf{M}$ is self-adjoint ($\mathbf{M}^\mathrm{T}=\mathbf{M}$), since a diagonal matrix of 0s and 1s is symmetric; applying it again simply re-zeros the same unacquired entries, which is clearly not the same as recovering the missing data (1 pt)
- General statement: adjoint ≠ inverse in general. $\mathbf{M}$ illustrates this clearly: it is self-adjoint, yet has no inverse at all, since it irreversibly zeros out the missing k-space entries; $\mathbf{F}$, by contrast, happens to have its adjoint equal its inverse (1 pt)

**Q3 (3 pts).** Why does applying $\mathbf{A}^\mathrm{T}\mathbf{d}$ (the adjoint of the whole system matrix) to the measured data not directly yield the reconstructed image?

*Anticipated points:*

- $\mathbf{A}^\mathrm{T}$ is not $\mathbf{A}^{-1}$; adjoint ≠ inverse, especially since $\mathbf{M}$ alone already has no inverse — it is a rank-deficient masking operator that irreversibly zeros out the missing k-space entries (1 pt)
- $\mathbf{A}^\mathrm{T}\mathbf{d}$ only produces a "regridded"/zero-filled image, still containing aliasing, not the true image (1 pt)
- Correctly connects this to needing an optimization/inversion step (e.g. solving a normal equation) to obtain the true reconstruction (1 pt)

---

### Card II-B: Least-Squares Reconstruction

**Q1 (4 pts).** State the least-squares cost function for MRI reconstruction and derive the normal equation from it. Why do we minimize a cost function instead of solving $\mathbf{d}=\mathbf{A}\mathbf{x}$ directly?

*Anticipated points:*

- Cost function $f(\mathbf{x}) = \tfrac{1}{2}\|\mathbf{A}\mathbf{x}-\mathbf{d}\|_2^2$ (1 pt)
- Setting the derivative to zero yields the normal equation $\mathbf{A}^\mathrm{T}\mathbf{A}\mathbf{x}=\mathbf{A}^\mathrm{T}\mathbf{d}$ (1 pt)
- Correctly states that $\mathbf{d}=\mathbf{A}\mathbf{x}$ is generally an inconsistent/overdetermined or ill-posed system with no exact solution, so we find a best fit instead (1 pt)
- States that $\mathbf{A}$ is generally not square/invertible, so direct inversion is not possible (1 pt)

**Q2 (3 pts).** Why is the least-squares cost function the "correct" choice under a statistical noise model, rather than an arbitrary choice?

*Anticipated points:*

- Assuming Gaussian measurement noise, maximizing the likelihood $P(\mathbf{d}_\mathrm{meas} \mid \mathbf{d}_\mathrm{true})$ is equivalent to minimizing $\|\mathbf{d}_\mathrm{meas}-\mathbf{d}_\mathrm{true}\|_2^2$ (1 pt)
- Because noise on each sample is independent, the joint probability factorizes into a product of per-sample Gaussians, which becomes a sum of squared errors after taking the log (1 pt)
- If not mentioned, probe: "What does a larger residual imply about the likelihood of that measurement?" — correct answer: a larger residual corresponds to a *lower* probability under the Gaussian model, not a higher one (1 pt)

**Q3 (3 pts).** What key mathematical properties does the least-squares cost function have, and what do they guarantee for optimization?

*Anticipated points:*

- Convex: has a single minimum that is also the global minimum, so gradient-based methods cannot get stuck in a local minimum (1 pt)
- Smooth/differentiable everywhere, which is what allows the gradient to be used as a search direction in iterative methods (1 pt)
- If not mentioned, probe: "Does convexity mean the minimum can only be reached by explicit matrix inversion?" — correct answer: no, convexity is exactly what guarantees iterative gradient-based methods can converge to the true minimum, without needing explicit inversion (1 pt)

---

### Card II-C: Gradient Descent

**Q1 (4 pts).** Derive the gradient of the least-squares cost function $f(\mathbf{x})=\tfrac{1}{2}\|\mathbf{A}\mathbf{x}-\mathbf{d}\|_2^2$, and write down the basic gradient descent update rule.

*Anticipated points:*

- Sets up the differentiation correctly, e.g. expanding $f(\mathbf{x})=\tfrac{1}{2}(\mathbf{A}\mathbf{x}-\mathbf{d})^\mathrm{T}(\mathbf{A}\mathbf{x}-\mathbf{d})$ before differentiating (1 pt)
- Arrives at the correct gradient $g(\mathbf{x}) = \mathbf{A}^\mathrm{T}(\mathbf{A}\mathbf{x}-\mathbf{d})$ (1 pt)
- States the update rule: $\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \eta\, g(\mathbf{x}^{(k)})$ (1 pt)
- If not mentioned, probe: "Why must $\mathbf{A}$ be transposed rather than applied directly?" — correct answer: the transpose is required to map the residual (which lives in data/k-space) back into image space, where $\mathbf{x}$ lives (1 pt)

**Q2 (3 pts).** What happens if the step size $\eta$ is too small, or too large?

*Anticipated points:*

- Too small: convergence becomes very slow, requiring many iterations to reach the minimum (1 pt)
- Too large: risk of overshooting the minimum, causing oscillation or even divergence (the cost increasing instead of decreasing) (1 pt)
- Correctly frames this as motivating the need for a "just right" step size — setting up why an optimal $\eta$ (derived in Q3) is worth having (1 pt)

**Q3 (3 pts).** Derive the analytically optimal step size $\eta$ at iteration $k$, and explain in what sense it is "optimal."

*Anticipated points:*

- Sets up the one-dimensional problem along the search direction: $\varphi(\eta) = f(\mathbf{x}^{(k)} - \eta\, g(\mathbf{x}^{(k)}))$ (1 pt)
- Differentiates with respect to $\eta$ and solves, arriving at $\eta = \dfrac{g(\mathbf{x}^{(k)})^\mathrm{T} g(\mathbf{x}^{(k)})}{g(\mathbf{x}^{(k)})^\mathrm{T} \mathbf{A}^\mathrm{T}\mathbf{A}\, g(\mathbf{x}^{(k)})}$ (1 pt)
- Explains the sense of optimality: this $\eta$ exactly minimizes the cost function along the current (negative gradient) search direction — it is the best possible single step in that direction, but does not by itself guarantee fast overall convergence on ill-conditioned problems (1 pt)

---

### Card II-D: Conjugate Gradient

**Q1 (4 pts).** What does it mean for an optimization problem to be "ill-conditioned" versus "well-conditioned"? Why does gradient descent zig-zag specifically on an ill-conditioned problem?

*Anticipated points:*

- Ill-conditioned ("valley"): the cost function has very different curvature along different directions, producing an elongated, valley-like shape (1 pt)
- Well-conditioned ("bowl"): curvature is similar in all directions, and gradient descent converges quickly and directly (1 pt)
- Explains that on an ill-conditioned problem, gradient descent zig-zags and needs many iterations, since the gradient direction is not aligned with the direction toward the minimum in an elongated valley (1 pt)
- Correctly distinguishes conditioning from convexity: convexity guarantees a unique global minimum exists, but conditioning is a separate property that governs how fast an iterative method actually reaches it (1 pt)

**Q2 (3 pts).** What is the conjugate gradient (CG) search direction, and why does incorporating the previous direction fix the zig-zagging described in Q1?

*Anticipated points:*

- $p(\mathbf{x}^{(k)}) = g(\mathbf{x}^{(k)}) + \beta(\mathbf{x}^{(k)})\, g(\mathbf{x}^{(k-1)})$, combining the current and previous gradient — unlike plain gradient descent, which uses only the current gradient direction, $-g(\mathbf{x}^{(k)})$ (1 pt)
- Explains the mechanism: blending in the previous direction lets the search "remember" where it just came from, so successive steps don't simply reverse each other across the valley (1 pt)
- Correctly connects this to Q1's diagnosis: the plain gradient direction alone was the cause of the zig-zag, and CG's fix is precisely to stop relying on that direction alone (1 pt)

**Q3 (3 pts).** Does conjugate gradient eliminate the need to compute a step size at each iteration? For the ill-conditioned example from the lecture, roughly how many iterations did gradient descent versus conjugate gradient each require?

*Anticipated points:*

- Correctly states that CG still requires computing a step size $\eta$ at each iteration — it does not eliminate step-size computation (1 pt)
- Recalls that gradient descent required roughly 21 iterations while CG required only about 2, for the illustrated ill-conditioned example (1 pt)
- Any correct reasoning for why a smarter search direction doesn't remove the need to determine how far to move along it (1 pt)

---

### Card II-E: Undersampling & Parallel Imaging

**Q1 (4 pts).** What happens when Cartesian k-space is undersampled by an acceleration factor $R$? What is gained and what artifact is introduced?

*Anticipated points:*

- Only every $R$-th phase-encoding line is acquired, reducing scan time by a factor of $R$ (1 pt)
- This causes $R$-fold aliasing, since the effective FOV in the phase-encoding direction is reduced by $R$ (1 pt)
- If not mentioned, probe: "Does the aliasing appear along the readout direction too?" — correct answer: no, only along the phase-encoding direction; the readout gradient/bandwidth is unaffected (1 pt)
- If not mentioned, probe: "Does undersampling improve SNR, since fewer lines are acquired?" — correct answer: no, undersampling typically reduces, not improves, SNR (1 pt)

**Q2 (3 pts).** How do multiple receiver coils and their sensitivity maps help resolve this aliasing?

*Anticipated points:*

- Each coil has a spatially varying sensitivity profile, weighting the signal differently depending on position (1 pt)
- This extra spatial information can be exploited to reconstruct an unaliased image from undersampled k-space data (1 pt)
- If not mentioned, probe: "Do all coils in the array measure essentially the same signal, and is this spatial information encoded the same way gradients encode it?" — correct answer: no, sensitivity profiles differ spatially between coils, and this is encoded through an amplitude weighting, not a linear phase ramp like gradients (1 pt)

**Q3 (3 pts).** Why does parallel imaging reconstruction quality degrade as the acceleration factor $R$ increases, even before $R$ reaches the number of coils $N_c$?

*Anticipated points:*

- Explains that as $R$ increases, the locations that alias onto the same pixel become spaced closer together in the object (1 pt)
- Connects this to coil sensitivity profiles varying smoothly in space, so closely-spaced locations have less distinct sensitivity values across coils — making the system of equations used to unfold them increasingly ill-conditioned (1 pt)
- Explains the consequence: solving an ill-conditioned system amplifies noise in the reconstructed image, so image quality degrades progressively as $R$ grows, well before $R$ reaches the hard limit of $N_c$ (1 pt)

---

## CATEGORY III — Regularization, Compressed Sensing & Learned Reconstruction

**How this category works:** drawn at the exam table, answered close to cold. The joker redraw applies to this category: one redraw, once, across Category II and III combined.

---

### Card III-A: Regularization & Compressed Sensing

**Q1 (4 pts).** Why is regularization needed for reconstructing highly undersampled MRI data, and what property should a regularization term $R(\mathbf{x})$ have?

*Anticipated points:*

- At low acceleration ($R\leq 2$) the least-squares minimum is unique and artifact-free, but from $R=4$ onward the minimum is no longer unique and does not correspond to the artifact-free image (1 pt)
- A regularization term $R(\mathbf{x})$ should assign low cost to artifact-free images and high cost to images containing aliasing artifacts (1 pt)
- If not mentioned, probe: "Does the normal equation always have a unique solution, regardless of acceleration factor?" — correct answer: no, at high acceleration the system becomes underdetermined and the solution is no longer unique (1 pt)
- If not mentioned, probe: "Does adding regularization only speed up convergence, without changing the final solution?" — correct answer: no, regularization changes which solution the optimization ultimately converges to (1 pt)

**Q2 (3 pts).** Explain Total Variation (TV) regularization: what does the gradient operator $\mathbf{G}$ compute, and why is the $L_1$-norm used instead of $L_2$?

*Anticipated points:*

- $\mathbf{G}$ computes the difference between adjacent pixels; sparse (mostly zero) for a fully-sampled, artifact-free image, and dense for an undersampled, aliased image (1 pt)
- The $L_1$-norm assigns a low cost to sparse gradients and a high cost to dense gradients, matching the goal of penalizing aliasing artifacts more than genuine image edges (1 pt)
- If not mentioned, probe: "Which of the two norms is non-differentiable?" — correct answer: $L_1$ is non-differentiable and requires a proximal operator; $L_2$ is smooth everywhere (1 pt)

**Q3 (3 pts).** Why is incoherence of undersampling artifacts important for compressed sensing, and why does this favor non-Cartesian trajectories like radial sampling?

*Anticipated points:*

- Compressed sensing relies on undersampling artifacts being incoherent, which is why non-Cartesian trajectories such as radial sampling are typically required, so the aliasing appears noise-like rather than as coherent ghosting (1 pt)
- Cartesian undersampling produces coherent, structured aliasing (e.g. overlapping mirrored copies of the image), which a sparsity-promoting regularizer cannot adequately resolve once the acceleration factor becomes too high (1 pt)
- If not mentioned, probe: "Does the choice of k-space trajectory actually matter for compressed sensing quality?" — correct answer: yes, it determines whether aliasing is coherent or incoherent, which directly affects whether TV/$L_1$ regularization can separate artifact from true signal (1 pt)

---

### Card III-B: ADMM

**Q1 (4 pts).** Write down the augmented Lagrangian for TV-regularized reconstruction and describe the three ADMM update steps at a high level.

*Anticipated points:*

- $L_\rho(\mathbf{x},\mathbf{z},\mathbf{u}) = \tfrac{1}{2}\|\mathbf{A}\mathbf{x}-\mathbf{d}\|_2^2 + \lambda\|\mathbf{z}\|_{1,2} + \tfrac{\rho}{2}\|\mathbf{G}\mathbf{x}-\mathbf{z}+\mathbf{u}\|_2^2$ (1 pt)
- x-update: minimize over $\mathbf{x}$ with $\mathbf{z},\mathbf{u}$ fixed → smooth quadratic problem (1 pt)
- z-update: minimize over $\mathbf{z}$ with $\mathbf{x},\mathbf{u}$ fixed → closed-form proximal/soft-thresholding step (1 pt)
- u-update: simple dual-variable update, $\mathbf{u}^{(k+1)}=\mathbf{u}^{(k)}+\mathbf{G}\mathbf{x}^{(k+1)}-\mathbf{z}^{(k+1)}$ (1 pt)

**Q2 (3 pts).** Why is the problem split into separate x- and z-updates rather than solved jointly in one step?

*Anticipated points:*

- In the x-update, the term $\lambda\|\mathbf{z}\|_{1,2}$ is held constant (z, u fixed) and drops out, leaving a smooth quadratic problem solvable via conjugate gradient (1 pt)
- In the z-update, the data-consistency term $\tfrac{1}{2}\|\mathbf{A}\mathbf{x}-\mathbf{d}\|_2^2$ is held constant (x fixed) and drops out, leaving a closed-form proximal/soft-thresholding solution (1 pt)
- If not mentioned, probe: "During the z-update, does the data-consistency term still depend on $\mathbf{x}$ through the constraint $\mathbf{G}\mathbf{x}=\mathbf{z}$?" — correct answer: no, $\mathbf{x}$ is held fixed during the z-update by construction, so that term is simply a constant, not something requiring joint minimization (1 pt)

**Q3 (3 pts).** Minimizing $L_\rho$ over $\mathbf{x}$ with $\mathbf{z}$ and $\mathbf{u}$ held fixed leaves a smooth quadratic problem. Derive the linear system that $\mathbf{x}^{(k+1)}$ must satisfy.

*Anticipated points:*

- Correctly drops the constant term $\lambda\|\mathbf{z}\|_{1,2}$, since it does not depend on $\mathbf{x}$ (1 pt)
- Differentiates the remaining terms and sets the result to zero: $\mathbf{A}^\mathrm{T}(\mathbf{A}\mathbf{x}-\mathbf{d}) + \rho\,\mathbf{G}^\mathrm{T}(\mathbf{G}\mathbf{x}-\mathbf{z}+\mathbf{u}) = \mathbf{0}$ (1 pt)
- Rearranges to the correct linear system: $(\mathbf{A}^\mathrm{T}\mathbf{A}+\rho\,\mathbf{G}^\mathrm{T}\mathbf{G})\,\mathbf{x}^{(k+1)} = \mathbf{A}^\mathrm{T}\mathbf{d} + \rho\,\mathbf{G}^\mathrm{T}(\mathbf{z}^{(k)}-\mathbf{u}^{(k)})$ (1 pt)

---

### Card III-C: Neural Network Building Blocks

**Q1 (4 pts).** A convolutional layer applies a small filter (e.g. 3×3) to an image. Write down a 3×3 filter that implements the same x-direction finite-difference gradient operation as the classical gradient operator $\mathbf{G}_x$ seen in TV regularization. How does this connect a learnable convolution to that fixed, hand-designed linear operator, and what role does the ReLU non-linearity play afterward?

*Anticipated points:*

- Correctly writes down a 3×3 filter computing the x-direction finite difference, e.g. rows `0 0 0` / `0 1 -1` / `0 0 0` (or an equivalent sign/orientation) (1 pt)
- Explains that this filter, applied via convolution, computes the same operation as the fixed gradient operator $\mathbf{G}_x$ from TV regularization — a convolution *is* a linear operator, just organized as a small sliding filter (1 pt)
- Correctly contrasts that in a CNN this filter's values are learned from data, rather than hand-designed as $\mathbf{G}_x$ is (1 pt)
- Explains the role of ReLU: introduces non-linearity after the (learnable) linear convolution step, analogous to how a soft-thresholding operator introduces non-linearity in a classical proximal step (1 pt)

**Q2 (3 pts).** What is the vanishing gradient problem in deep networks, and why does it occur?

*Anticipated points:*

- In very deep networks, gradients computed via backpropagation can become extremely small by the time they reach early layers (1 pt)
- Consequently, those early layers receive very little training signal and learn very slowly (1 pt)
- If not mentioned, probe: "Is this about layers learning too quickly, or too slowly?" — correct answer: too slowly (or barely at all); it is unrelated to overfitting (1 pt)

**Q3 (3 pts).** How do residual networks (ResNets) address the vanishing gradient problem, and what exactly changes in the network?

*Anticipated points:*

- A skip (residual) connection provides a direct path for the gradient to flow backward through the network (1 pt)
- This mitigates the vanishing gradient problem without changing the overall depth of the network (1 pt)
- If not mentioned, probe: "Do skip connections work by removing layers from the network?" — correct answer: no, the network depth is unchanged; skip connections simply provide an additional, shorter path for the gradient (1 pt)

---

### Card III-D: Physics-Based Hybrid Networks (Deep ADMM)

**Q1 (4 pts).** In classical ADMM with hand-crafted total-variation regularization, the z-update has a closed-form solution using soft-thresholding. State this classical update. What replaces this step in "Deep ADMM," and how does its role within the overall algorithm stay the same despite this change?

*Anticipated points:*

- States the classical z-update: $\mathbf{z}^{(k+1)}=\mathrm{S}_{\lambda/\rho}(\mathbf{G}\mathbf{x}^{(k+1)}+\mathbf{u}^{(k)})$ (1 pt)
- In Deep ADMM, this z-update is replaced by a learned residual network, $\mathrm{ResNet}(\mathbf{x}^{(k+1)}+\mathbf{u}^{(k)},\boldsymbol{\theta})$ (1 pt)
- Explains that the role stays the same: it is still a proximal/denoising step that takes $\mathbf{x}^{(k+1)}+\mathbf{u}^{(k)}$ as input and produces the next $\mathbf{z}$, just computed by a network instead of a formula (1 pt)
- If not mentioned, probe: "Does this mean the whole ADMM iterative structure changes, or just this one internal step?" — correct answer: only this internal step changes; the overall alternating x/z/u structure is unchanged (1 pt)

**Q2 (3 pts).** In the standard ADMM x-update, the gradient operator $\mathbf{G}$ appears (via $\mathbf{G}^\mathrm{T}\mathbf{G}$ in the linear system solved by conjugate gradient). What happens to $\mathbf{G}$ in Deep ADMM's x-update, and which parameters overall become learnable rather than fixed by hand?

*Anticipated points:*

- The gradient operator $\mathbf{G}$ is replaced by the identity operator $\mathbf{I}$ (1 pt)
- Learnable parameters: $\rho$ and the network parameters $\boldsymbol{\theta}$ (1 pt)
- If not mentioned, probe: "Do the u-update or the conjugate-gradient-based structure of the x-update also get replaced by a network?" — correct answer: no, the x-update and u-update retain their original algebraic form; only the z-update becomes a trainable network (1 pt)

**Q3 (3 pts).** Why keep the x-update (data consistency) physics-based/model-driven instead of learning the entire reconstruction end-to-end with a single large network?

*Anticipated points:*

- The x-update enforces consistency with the actual measured data via the known forward model $\mathbf{A}$, which a purely learned network has no guarantee of respecting (1 pt)
- This hybrid structure combines the reliability of a physics-based data-consistency term with the flexibility of a learned prior, typically needing less training data than a fully learned end-to-end mapping (1 pt)
- Any reasonable, correctly-reasoned articulation of the trade-off between fully learned and hybrid physics-informed approaches (1 pt)

---

### Card III-E: Training Strategies

**Q1 (4 pts).** Write down the loss function used for supervised training of a deep-learning MRI reconstruction network, and explain what each term represents.

*Anticipated points:*

- Writes down (or an equivalent notation for) the loss: $L(\boldsymbol{\theta}) = \|\mathbf{M}_\mathrm{FS}\mathbf{F}\mathbf{C}\cdot\mathrm{Network}(\mathbf{A}_\mathrm{US}^\mathrm{T}\mathbf{d}_\mathrm{US},\boldsymbol{\theta}) - \mathbf{d}_\mathrm{FS}\|$ (1 pt)
- Correctly identifies $\mathbf{A}_\mathrm{US}^\mathrm{T}\mathbf{d}_\mathrm{US}$ as the zero-filled/regridded image built from the undersampled measurement, fed into the network as input (1 pt)
- Correctly identifies that the network's reconstructed image is re-encoded via $\mathbf{M}_\mathrm{FS}\mathbf{F}\mathbf{C}$ (i.e. without undersampling) and compared against $\mathbf{d}_\mathrm{FS}$, the actually measured fully sampled k-space (1 pt)
- If not mentioned, probe: "Could $\mathbf{d}_\mathrm{US}$ and $\mathbf{d}_\mathrm{FS}$ just be the same measurement?" — correct answer: no, this requires two genuinely different acquisitions of the same scan — an undersampled one to build the input and a fully sampled one to serve as the target (1 pt)

**Q2 (3 pts).** What is a practical limitation of supervised training in real clinical settings, and what problem does it motivate?

*Anticipated points:*

- Fully sampled reference data is often difficult, slow, or impossible to acquire in practice (e.g. due to patient motion, long scan times, or the very reason undersampling is used in the first place) (1 pt)
- Correctly identifies that this motivates finding a way to train without any fully sampled reference (1 pt)
- Any correct additional reasoning about why this is a fundamental, not incidental, limitation of the supervised approach (1 pt)

**Q3 (3 pts).** Explain zero-shot self-supervised learning: how is training performed without fully sampled data, what does the loss function look like, and in what sense is it "zero-shot"?

*Anticipated points:*

- Measured undersampled k-space is repeatedly split into two disjoint sets A and B; the network is trained to predict set B from set A as input (1 pt)
- Writes down (or an equivalent notation for) the loss: $L(\boldsymbol{\theta}) = \|\mathbf{M}_B\mathbf{F}\mathbf{C}\cdot\mathrm{Network}(\mathbf{A}_A^\mathrm{T}\mathbf{d}_A,\boldsymbol{\theta}) - \mathbf{d}_B\|$ — structurally identical to Q1's supervised loss, but with subsets $A$/$B$ of the same undersampled acquisition standing in for the undersampled/fully-sampled pair (1 pt)
- Because $A$ and $B$ both come from the same undersampled scan, no fully sampled reference is ever needed — training and inference can be performed on a single dataset, which is why it is called "zero-shot" (1 pt)

---

## Summary Table

| Category | Card | Topic |
|---|---|---|
| I | A | Cartesian GRE (notebook + debrief) |
| I | B | EPI (notebook + debrief) |
| I | C | Radial GRE (notebook + debrief) |
| II | A | Linear Forward Model & Adjoints |
| II | B | Least-Squares Reconstruction |
| II | C | Gradient Descent |
| II | D | Conjugate Gradient |
| II | E | Undersampling & Parallel Imaging |
| III | A | Regularization & Compressed Sensing |
| III | B | ADMM |
| III | C | Neural Network Building Blocks |
| III | D | Physics-Based Hybrid Networks (Deep ADMM) |
| III | E | Training Strategies |