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

## N13. Corruptions
* `imagecorruptions` is applied to the original-resolution test image; images with a side < 32 px are first
  upsampled (bilinear) so that min side = 32, because several corruptions assume at least 32 px. Any image on
  which the package fails falls back to the ImageNet-C formula implemented in `src/corruptions.py`; such events
  are appended to `OUT/data_cache/gtsrb_c/fallbacks.txt` and must be copied here.
* Cache size: 30 uint8 arrays × 12,630 × 3 × 32 × 32 ≈ 1.1 GB on Drive (regenerated if absent).

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
