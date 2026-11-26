# NOTES — adaptations and paper/code discrepancies

Every entry here is a choice the student must be able to justify (CLAUDE.md tags: `[adaptation]`, or a
paper-vs-code difference resolved in favour of the official code, §5.7).

## N1. DAT generator architecture (official code vs CLAUDE.md §5.7)
* **Code:** `Generator_MLP` has **five** linear layers (128 → 256 → 512 → 1024 → 3·H·W), LeakyReLU(0.2), with
  `BatchNorm1d(out, 0.8)` after blocks 2–4 (the positional `0.8` is the `eps` argument of `BatchNorm1d`; it is
  reproduced as eps = 0.8). CLAUDE.md said "four linear layers"; the code (4 hidden blocks + output layer) is followed.
* Input = `z ~ N(0, I)` with 100 dims concatenated with the classifier **logits** of the clean image (143 dims).
* Output = sigmoid → values in (0, 1), reshaped to **3 × 32 × 32** and mixed with the raw fftshift-ed amplitude
  with **no rescaling**: `A_mix = λ·A_G + (1 − λ)·A`. Since the raw amplitude is O(1)–O(1000) while A_G ∈ (0,1),
  the mix mostly attenuates the amplitude by (1 − λ) plus a small learned term. This is what the official code
  does and is reproduced as-is (file `src/aag.py`, `src/methods/dat.py`).
* The official module also defines an unused `nn.Embedding` and `nn.Linear`; omitted.

## N2. DAT spectrum convention (code vs CLAUDE.md §5.7)
* CLAUDE.md wrote `rfft2` (3×32×17). The official code uses the **full** `fft.fftn` over (H, W), fftshift-s the
  amplitude only, keeps the phase unshifted, and reconstructs with `ifftshift(A)·exp(iP) → ifftn → real → clip[0,1]`.
  The code is followed (`src/fourier.py: fft2_dat / ifft2_dat`); the generator therefore outputs 3×32×32 and the
  output **is clipped** to [0, 1] (the code does).

## N3. DAT branch inputs (code)
* The code computes amplitude/phase and the conditioning logits from the **un-augmented** clean image (`img_o`,
  `transform_test`) and applies `RandomHorizontalFlip + RandomCrop(32, padding=3)` **after** the mix (kornia).
  Here: the dataset returns `x_orig`; the DAT branch mixes `x_orig` and then applies the project's random crop
  (32, padding 3, reflect) — **no flip** (CLAUDE.md §3.2). Rotation / colour jitter are not applied post-mix
  (the code does not either). The base branch uses the normally augmented `x`.

## N4. Label smoothing 0.5 in the shared adversarial base (official DAT code) — DECIDED 2026-11-14
* The official code sets `args.alpha = 0.5` and uses it as `label_smoothing=0.5` in **every** CE term: in the
  EAEG attack loss (`perturb_input`) and in both branches of the training loss (`one_step`, `criterion`).
* CLAUDE.md §4.5 writes `CE(f(v), y)` plainly, but §5.7 says to follow the code where paper and code disagree,
  and the fairness rule requires E3/E5/E6 to share one base. **Student decision (2026-11-14): follow the code.**
  `configs/_defaults.yaml` sets `label_smoothing: 0.5`; it is consumed only by the shared base, i.e. the EAEG
  attack CE and the L_AT CE of E3, E5, E6, A1–A6, and the plain CE that replaces L_AT in E4.
* E0 (standard CE), E1 (PGD-AT) and E2 (TRADES) do **not** use label smoothing, so they match their papers.
  Consequence for the thesis: part of any E3-vs-E2 gap may come from label smoothing rather than from EAEG;
  say so when discussing H1.

## N5. KL direction / reduction / detach (code, CLAUDE.md §4.5)
* Attack and loss β-terms use `F.kl_div(log_softmax(f(v')), softmax(f(v)), reduction='batchmean')`
  = KL( f(v) ‖ f(v') ). In the training loss neither side is detached (code). In the attack, f(v) is a
  constant w.r.t. v' (computed once under `no_grad`, equivalent to the code which recomputes it each step).
* JS term exactly as in the code: `m = clamp((p+q)/2, 1e-7, 1).log(); JS = (kl_div(m, p) + kl_div(m, q)) / 2`.

## N6. DAT generator step
* As in the code, the generator is updated **after** the classifier step with a **fresh forward pass** of all
  four inputs (x, x', x̂ attached, x̂') and `g_loss = −L_DAT`; BN running statistics are therefore updated twice
  per batch (code behaviour). The classifier's parameters are frozen (`requires_grad=False`) during this pass
  purely to save compute; in the code their gradients are simply discarded by the next `zero_grad()`.
* AWP (`awp_warmup=10000`) and the weight-averaging variables in the official script are inactive by default and
  are not ported. The code's 200-epoch schedule is replaced by the shared schedule (§4.3).

## N7. Fo-Mixup segment partition on the Nyquist row/column [adaptation]
* With a 32-point FFT the row/column at shifted index 0 (du = −16 or dv = −16) is its own conjugate mirror, so
  `atan2(dv, du) mod π` would place (−16, 5) and (−16, −5) in different sectors and break conjugate symmetry.
  On those positions the angle is computed from (|du|, |dv|). All 144-segment masks are verified
  conjugate-symmetric by `tests/test_fourier_fomixup.py`.
* Empty (ring, sector) combinations (e.g. inner rings have fewer than 12 sectors) are excluded from K-means
  and stored with cluster label 255.

## N8. Fo-Mixup Eq. 2 noise p [adaptation]
* `p ~ N(1, 0.1²)` is sampled **conjugate-symmetrically** (one half of the spectrum sampled, mirrored onto the
  other) so the mixed spectrum stays Hermitian and x̃ is real. Taking the real part of a non-symmetric spectrum
  is mathematically the same as symmetrising the spectrum, so this only makes the implementation explicit.

## N9. Confidence sampling implementation [adaptation]
* The entropy of f(x̃) is computed with BN-B in **eval** mode under `no_grad` (so BN-B running statistics are not
  polluted by an extra forward), then only the kept 50% of x̃ are attacked and forwarded through BN-B in train
  mode. This equals "mask and renormalise by the number kept" for every x̃ loss term, while BN-B batch
  statistics are computed on the kept subset only, and it halves the x̃-branch attack cost.
* The consistency KL (Eq. 3) is implemented literally as KL( σ(f(x̃)/τ) ‖ σ(f(x)/τ) ) with τ = 4, weight 1.0,
  no τ² factor, both sides attached.

## N10. K-means verification
* The test compares the batched torch K-means with scikit-learn's Lloyd algorithm started from the **same quantile
  centres** (`n_init=1, max_iter=20, tol=0`), i.e. it verifies the implementation, not k-means++ initialisation.

## N11. train_log.csv `train_clean_acc` for E1
* PGD-AT (E1) forwards only x′ during training; its logged training accuracy is measured on x′. All other
  methods log accuracy on the clean x (BN-A).

## N12. AutoAttack protocol
* `version='standard'` attacks (apgd-ce, apgd-t, fab-t, square) are each run **alone** via
  `attacks_to_run=[atk]` so that a standalone per-attack accuracy exists (needed for the Square sanity check,
  §7.1). The overall AutoAttack accuracy is the fraction of images robust to **all four** — the same quantity
  the sequential 'standard' run reports. Each attack's result is check-pointed (`eval_flags.pt`).

## N13. Corruptions implemented directly (the `imagecorruptions` package is broken on current Colab)
* `imagecorruptions` 1.1.2 (last release) calls `skimage.filters.gaussian(..., multichannel=True)`, removed in
  scikit-image 0.19, imports `pkg_resources`, and depends on `opencv-python`, which conflicts with Colab's
  preinstalled OpenCV. CLAUDE.md §3.3 allows implementing the corruptions directly when the package fails.
* `src/corruptions.py` implements the six corruptions with the package's formulas and severity constants
  (Hendrycks & Dietterich, ICLR 2019). Gaussian blur uses `scipy.ndimage.gaussian_filter` with
  `mode='nearest', truncate=4`, which is what `skimage.filters.gaussian` computes per channel; brightness uses
  matplotlib's RGB↔HSV, the same formulas as `skimage.color`.
* Corruptions are applied to the original-resolution test image (no upsampling of small images any more), then
  resized to 32×32. Randomness: one `np.random.RandomState(1000·corruption_index + severity)` per array.
* Cache: 30 uint8 arrays × 12,630 × 3 × 32 × 32 ≈ 1.1 GB, synced to storage and fetched on demand.
* Smoke runs keep their tiny synthetic cache under `OUT/.smoke/`, and a cache of the wrong size is rebuilt.

## N14. Patch attack details [adaptation]
* The patch is initialised at the image's own pixels (so the attack can only lower accuracy from clean), then
  100 signed-gradient steps of 0.05 in [0, 1]. Location: top-left corner uniformly inside the central 24×24
  region, `np.random.RandomState(image_index)`.

## N15. Model initialisation
* All methods use the DAT code's initialisation (Kaiming fan-out conv, Xavier linear, BN γ=1 β=0).

## N16. `--epochs` / `--max-batches` flags
* Exist only for the resume test and smoke runs. Real runs must use the shared schedule in `_defaults.yaml`.

## Open decisions for the student
1. ~~N4 label smoothing~~ decided: 0.5 (2026-11-14).
2. §4.3: 60 vs 100 epochs, decided at Milestone 2 from E3's validation PGD-20 curve.

## N17. Numerically safe KL / JS (value identical to the official code)
* `F.kl_div(log_q, p)` with `p = softmax(logits)` as a non-detached target has a NaN gradient when some `p_c`
  underflows to exactly 0 (the `xlogy` backward evaluates 0/0). This was reproduced in a CPU smoke run of E6
  (generator weights became NaN in one step while gradients were still ~1e-3). All KL/JS terms are therefore
  computed as `sum_c p_c (log p_c − log q_c)` with `log p = log_softmax(logits)`, which has the same value and a
  finite gradient; the KL direction, reduction and JS mixture clamp (1e-7) of the DAT code are unchanged.

## N18. Dataset archive: official 39,209-image training set, not torchvision's archive
* `torchvision.datasets.GTSRB` downloads `GTSRB-Training_fixed.zip`, which holds **26,640** training images
  (the online-competition training set). CLAUDE.md §3 requires the official **39,209**-image final training set,
  which is why the v1 notebook stopped with an `AssertionError` in `make_split`.
* `src/data.py` now downloads the official archives from the same server torchvision uses
  (`GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip`, `GTSRB_Final_Test_GT.zip`), checks their
  MD5 sums and the image counts (39,209 / 12,630), and fails with an explanation otherwise. Checked on
  2026-11-20 by downloading all archives and counting the images.
* The 32×32 caches carry a version number; a cache built from the old archive is detected and rebuilt.

## N19. Colab packages
* `torchattacks` was removed: it is never imported, and its metadata pins `requests~=2.25.1`, which conflicts
  with the `requests` version Colab's own packages require (the v1 install-cell conflict).
* `imagecorruptions` was removed (N13).
* On Colab only AutoAttack is installed, pinned to commit a3922004 and with `--no-deps`, so no preinstalled
  package (torch, numpy, …) is changed (`requirements-colab.txt`). CLAUDE.md §2.3 lists the extras; the two
  removals are deviations from that list.

## N20. Results storage: local disk + sync (instead of writing to Google Drive directly)
* Training writes to the local VM disk (`/content/out`). `src/storage.py` uploads new or changed result files to
  persistent storage at most every 10 minutes during training and after every step, and restores them in a new
  session. Checkpoints and corruption arrays are fetched only when a step needs them.
* Preferred backend: a private Hugging Face dataset repository (free account, `HF_TOKEN` Colab secret). Fallback:
  a Google Drive folder, chosen automatically when Hugging Face is not configured or fails.
* At most `OUT_SYNC_INTERVAL` (10 minutes) of training is lost on a disconnect; `last.pt` is still written every
  epoch locally. The Hugging Face history is squashed after each run so superseded checkpoints do not
  accumulate.
* This changes only where files are kept, not any result; CLAUDE.md §8.1's layout is unchanged inside OUT.

## N21. Observation at Milestone 1: E0 keeps about 10-17% robust accuracy at ε = 8/255
* E0 seed 0 (T4, 2026-11-22): clean 99.32, FGSM 60.51, PGD-20 17.09, PGD-100 14.40, CW∞ 14.71,
  AutoAttack 10.35 (n = 2,000; APGD-CE 12.50, APGD-T 10.55, FAB-T 11.25, Square 25.70). ε sweep:
  75.64 / 46.23 / 17.14 / 6.20 / 2.26 at ε = 2 / 4 / 8 / 12 / 16 (/255).
* CLAUDE.md §12 expects "PGD-20 near 0%". The value is above that. This is recorded as a result; no setting
  was changed for it.
* It does not look like gradient masking: all §7.1 checks pass, the margin-based attacks (CW∞, APGD-T, FAB-T)
  agree with CE-PGD, the black-box Square attack is much weaker than the white-box attacks, and accuracy goes
  to about 2% at ε = 16/255. The attack code was re-read: float32, eval mode, BN-A, projection then clamp.
* A likely reason is that many GTSRB images are very dark: for pixels near 0, the [0, 1] clamp removes half of
  the ε-ball, so the effective budget is smaller than on CIFAR-10. This has not been tested.
* Consequence for the thesis: the robust accuracies of E1-E6 should be read against E0's 10.35% AutoAttack
  baseline, not against 0%.

## N22. Decisions at Milestones 2-3: 60 epochs and float32 training attacks for every run
* **Epochs.** The student kept the 60-epoch schedule (LR ×0.1 at 40 and 50) for all experiments, for the Colab
  compute budget (on a T4, one adversarial run plus its evaluation takes about 3-4 h for E3 and more for E5/E6).
  E0, E3 and E5 (seed 0) were trained with it before the decision, so all runs share one schedule.
  The §4.3 check (is E3's validation PGD-20 still clearly rising at epoch 50?) has not yet been reviewed.
  E3's best epoch is 54, just after the second LR drop. If the curve shows a clear rise at epoch 50, this is
  reported as a limitation of the study rather than changed.
* **Attack precision.** The attacks used during training (EAEG, PGD-10, TRADES inner maximisation) run in
  float32 with autocast disabled, like evaluation. Only the outer training step uses fp16 autocast (§4.3).
  Running the training attacks in fp16 was considered for speed (~1.5-2× estimated) but not adopted, because E3
  and E5 had already been trained in float32 and every method must use the same setup.
