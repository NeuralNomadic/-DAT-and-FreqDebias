# Project: Frequency-Domain Adversarial Robustness for Traffic-Sign Recognition
## FreqDebias-style dominant-band mixing vs DAT, on GTSRB, run on Google Colab

You are Claude Code, helping an M.Tech student build and run this study. Read the whole file before writing code. Work milestone by milestone (Section 12) and **stop after each milestone** for the student's review.

Every hyperparameter below is tagged:
- **[paper: X]** — taken from paper X.
- **[adaptation]** — a choice made for this project (GTSRB, 32×32, adversarial training, Colab). The student must be able to justify these, so record any change to them in `NOTES.md`.

---

## 1. Research questions and hypotheses

**Main question.** Two methods perturb an image's Fourier *amplitude* (style) while keeping its *phase* (shape), and both add a consistency loss. Which perturbation makes a traffic-sign classifier more robust?

- **Arm A — FreqDebias-AT:** amplitude mixing restricted to the frequency bands a standard model relies on most (Fo-Mixup, Kashiani et al., CVPR 2025), adapted to adversarial training.
- **Arm B — DAT:** amplitude generated adversarially by a learned generator over the whole spectrum (Li et al., NeurIPS 2024).

Both arms sit on the **same adversarial-training base** (Section 4.5), so the comparison isolates *how the amplitude is perturbed*.

**Hypotheses**
- **H1:** Both frequency arms improve robust accuracy over the shared base (E3) and over TRADES (E2).
- **H2:** Fo-Mixup *without* adversarial training (E4) gives little adversarial robustness — FreqDebias was designed for generalization, not adversarial robustness.
- **H3:** DAT wins on adversarial attacks (AutoAttack); FreqDebias-AT does at least as well on natural corruptions. (Open — this is what the study measures.)
- **H4:** Targeting dominant bands beats mixing the whole spectrum (ablation A2 vs A1 in Section 5.8).

---

## 2. Requirements

### 2.1 Accounts and storage
- Google account with **Google Colab** and **Google Drive**, at least **5 GB free on Drive** (about 25 runs × ~135 MB of checkpoints, plus logs and cached masks).
- A **GitHub** repository for the code. If private, the student stores a GitHub personal access token in **Colab Secrets** as `GITHUB_TOKEN`.

### 2.2 Hardware
- Colab GPU runtime. Free tier is usually a **T4 (16 GB)**; L4 or A100 on paid tiers. The code must run on any of them.
- Free sessions can disconnect after several hours; GPU access is not guaranteed. **All training is resumable** (Section 10.3).

### 2.3 Software
- Use the **PyTorch and torchvision preinstalled on Colab**. Do not reinstall or pin torch.
- `requirements.txt` (extras only):
  ```
  torchattacks
  git+https://github.com/fra31/auto-attack
  imagecorruptions
  scikit-learn
  pandas
  matplotlib
  pyyaml
  tqdm
  ```
- After the first successful Colab run, save `pip freeze` to `OUT/env/requirements.lock.txt` and record `torch.__version__`, `torchvision.__version__`, Python version, and GPU name in every `metrics.json`. This is the exact environment record for the thesis.

### 2.4 External code
- DAT official code: https://github.com/Feng-peng-Li/DAT — port the generator and loss details from it (Section 5.7). Check its license and credit it in `README.md`.
- FreqDebias has no code to port; implement Fo-Mixup from its paper exactly as specified in Section 6.

---

## 3. Dataset: GTSRB

| Item | Value |
| --- | --- |
| Source | `torchvision.datasets.GTSRB(root, split='train'/'test', download=True)` |
| Classes | 43 |
| Official train | 39,209 images |
| Official test | 12,630 images (used **only** for final evaluation) |
| Input size | resize to **32×32**, bilinear, antialias=True [adaptation] |
| Pixel range | **[0, 1]**, float32 |

### 3.1 Train/validation split — by track [adaptation]
- Training filenames are `TTTTT_IIIII.ppm`; `TTTTT` is the track ID (≈30 consecutive photos of one physical sign).
- Per class, hold out **10% of tracks** (at least 1) for validation, chosen with seed 1234. **No track may appear in both splits.**
- Save the split once to `OUT/split/train.txt` and `OUT/split/val.txt`. Every experiment loads these files; none recomputes the split.

### 3.2 Augmentation (training only) [adaptation]
- `RandomCrop(32, padding=3, padding_mode='reflect')`
- `RandomRotation(10)`
- `ColorJitter(brightness=0.2, contrast=0.2)` applied with probability 0.5
- **No horizontal or vertical flips, ever.** Flipping changes sign meaning ("turn left" ↔ "turn right").
- No augmentation for validation or test.

### 3.3 Corrupted test set (for generalization)
- Six corruptions × severities 1–5: Gaussian noise, Gaussian blur, brightness, contrast, fog, JPEG compression.
- Apply each corruption to the **original-resolution** test image, then resize to 32×32 [adaptation].
- Use the `imagecorruptions` package; if any corruption errors or looks wrong, implement it directly and note it in `NOTES.md`.
- Deterministic: seed = 1000·corruption_index + severity. Cache to `OUT/data_cache/gtsrb_c/` as uint8 `.npy` (optional; regenerate if absent).
- Save a visual grid (5 signs × 5 severities per corruption) to `OUT/figures/corruptions/` for the student to inspect.

---

## 4. Shared setup (identical for every experiment)

### 4.1 Model
- ResNet-18, **CIFAR variant**: first conv 3×3, stride 1, padding 1; no max-pool; 43-way linear head after global average pooling. Final feature map for a 32×32 input: 512×4×4.
- **Normalization is the model's first layer** (per-channel mean/std computed once on the training split and saved to `OUT/split/stats.json`). Attacks therefore act on [0, 1] images.
- **Dual BatchNorm** option: every BN layer has two parameter sets, `A` and `B`, selected by a flag on the forward call. Used by E4, E5, E6 (Section 4.5). Test-time inference always uses **BN-A**.

### 4.2 Threat model
- L∞, **ε = 8/255**, step **α = 2/255**, random start. [paper: DAT; Madry]

### 4.3 Optimization [adaptation, unless tagged]
| Setting | Value |
| --- | --- |
| Optimizer | SGD, momentum 0.9, weight decay 5e-4 [paper: DAT] |
| Initial LR | 0.1 [paper: DAT] |
| Batch size | 128 |
| Epochs | **60**, LR ×0.1 at epochs 40 and 50 |
| Mixed precision | fp16 `autocast` + `GradScaler` for training; **all FFT and amplitude/phase arithmetic in float32** (autocast disabled there); evaluation always float32 |

If at Milestone 2 the validation PGD-20 accuracy of E3 is still clearly rising at epoch 50, change epochs to 100 (LR drops at 75 and 90) **for every experiment**, and record this in `NOTES.md`. Never give different methods different schedules.

### 4.4 Model selection
- Every 2 epochs, compute **PGD-20 accuracy on the full validation split** (ε = 8/255, α = 2/255, BN-A).
- `best.pt` = checkpoint with the highest validation PGD-20 accuracy. For E0 and E4 (no adversarial training), select on validation **clean** accuracy.
- Reported results always come from `best.pt`, never from the last epoch.

### 4.5 Shared adversarial-training base (used by E3, E5, E6)

**Efficient attack, EAEG** [paper: DAT, Alg. 1]. For input v with label y:
```
v' ← v + 0.001·N(0, I)
repeat K = 5 times:
    L_AE = CE(f(v'), y) + β·KL(f(v'), f(v))
    v'   ← clip_[0,1]( Proj_{ε-ball around v}( v' + α·sign(∇_{v'} L_AE) ) )
```
β = 15 [paper: DAT]. Generate attacks with the model in eval mode, then return to train mode. Use the KL direction, reduction, and detach placement from the DAT official code.

**Base adversarial loss** [paper: DAT, Eq. 6 definition of L_AT]:
```
L_AT(v) = CE(f(v), y) + β·KL(f(v'), f(v)),   v' = EAEG(v)
```

**Fairness rule:** E3, E5 and E6 use exactly this EAEG and L_AT, with the same K, β, ε, α. They differ only in the second branch (Section 5).

### 4.6 Seeds and reproducibility
- Seeds 0, 1, 2 for E1, E2, E3, E5, E6. Seed 0 only for E0, E4 and all ablations.
- Seed Python, NumPy and torch (CPU and CUDA). Enable `torch.backends.cudnn.deterministic = True` only in the resume test (Section 11); leave default for speed elsewhere.

---

## 5. Experiments

| ID | Name | Adversarial training | Second branch | Seeds |
| --- | --- | --- | --- | --- |
| E0 | Standard | none | none | 0 |
| E1 | PGD-AT | PGD-10, CE on x' only [paper: Madry] | none | 0, 1, 2 |
| E2 | TRADES | PGD-10 on KL, β = 6 [paper: TRADES] | none | 0, 1, 2 |
| E3 | Base-AT (control) | Section 4.5, L = L_AT(x) | none | 0, 1, 2 |
| E4 | FreqDebias-Clean | none | Fo-Mixup + consistency | 0 |
| E5 | **FreqDebias-AT (Arm A)** | Section 4.5 | Fo-Mixup + consistency | 0, 1, 2 |
| E6 | **DAT (Arm B)** | Section 4.5 | AAG + JS consistency | 0, 1, 2 |
| A1–A6 | Ablations | see 5.8 | | 0 |

### 5.1 E0 — Standard
`L = CE(f(x), y)`.

### 5.2 E1 — PGD-AT [paper: Madry]
Inner: PGD, K = 10, α = 2/255, uniform random start in the ε-ball, maximize CE. Outer: `L = CE(f(x'), y)`.

### 5.3 E2 — TRADES [paper: TRADES]
Inner: K = 10 steps maximizing `KL(f(x') ‖ f(x))`, start `x + 0.001·N(0, I)`. Outer: `L = CE(f(x), y) + 6·KL(f(x') ‖ f(x))`.

### 5.4 E3 — Base-AT (control)
`L = L_AT(x)` from Section 4.5. This is DAT's own ablation baseline [paper: DAT, Table 4] and the common base of both arms.

### 5.5 E4 — FreqDebias-Clean (tests H2)
Same as E5 but with **no attacks**: replace every `L_AT(v)` by `CE(f(v), y)`.

### 5.6 E5 — FreqDebias-AT (Arm A)
Per batch (x, y):
1. Build x̃ = **Fo-Mixup(x)** (Section 6). Partner image: a random training image **of the same class** [adaptation: FreqDebias mixes two forgery images, i.e. the same class].
2. **Confidence sampling** [paper: FreqDebias, λ = 0.5]: with `torch.no_grad()` and BN-B, compute the prediction entropy of f(x̃). Keep the 50% of the batch with the **lowest** entropy; the rest are excluded from every x̃ term below (mask and renormalize by the number kept). [adaptation: applied per batch rather than to a precomputed pool]
3. Loss:
   ```
   L_FD = 0.5·( L_AT(x) + L_AT(x̃) )
        + 1.0 · KL( σ(f(x̃)/τ) ‖ σ(f(x)/τ) )         # τ = 4  [paper: FreqDebias Eq. 3]
        + δ   · L_att(x, x̃)                           # δ = 0.1 [paper: FreqDebias Eq. 13]
   ```
   - x and x' use BN-A; x̃ and x̃' use BN-B [adaptation: matches DAT for a fair comparison].
   - **L_att (CAM consistency)** [adaptation of FreqDebias Eq. 4–6]: for the true class y, `CAM(v) = W_y^T · F(v)` where `F(v)` is the 512×4×4 final feature map and `W_y` the head's weight row for class y. Flatten to 16 values, apply `softmax(CAM/τ)` with τ = 4, and take the **Jensen–Shannon divergence** between x and x̃. (FreqDebias's separate CAM branch, K-means region mask and instance norm are omitted; for a ResNet with GAP + linear head the CAM is directly available.)
   - The vMF hypersphere consistency of FreqDebias is **not** used in the main arm [adaptation: a 43-class vMF classifier needs Bessel functions in 512 dimensions, which is numerically fragile]. It may be added later as an extension only if the student asks.

### 5.7 E6 — DAT (Arm B) [paper: DAT, Alg. 2; port details from official code]
Per batch (x, y):
1. `x' = EAEG(x)` (BN-A).
2. **Stage I.** `X = rfft2(x)` per channel (float32); `A = |X|` (shape 3×32×17), `P = angle(X)`. Generator input: `z ~ N(0, I)` with τ = 100, concatenated with f(x) logits (43) → 143 dims. Generator: **four linear layers**, sigmoid output, reshaped to 3×32×17. **Take hidden widths and the output's scaling relative to A from the official code.** Mix `A_mix = λ·A_G + (1 − λ)·A`, `λ ~ U(0, 1)`. `x̂ = irfft2(polar(A_mix, P), s=(32, 32))`; clamp only if the official code does.
3. `x̂' = EAEG(x̂)` (BN-B).
4. **Stage III.** `L_DAT = 0.5·(L_AT(x) + L_AT(x̂)) + 2·JS(f(x), f(x̂))` (ω = 2 [paper: DAT]). Classifier: SGD step that **decreases** L_DAT. Generator: its own SGD (lr 0.1, momentum 0.9, weight decay 5e-4 [paper: DAT]) that **increases** L_DAT (negated gradients or gradient-reversal layer — follow the official code).

If paper and code disagree, follow the code and write the difference in `NOTES.md`.

### 5.8 Ablations (seed 0)
| ID | Change | Question answered |
| --- | --- | --- |
| A1 | E5 with **B = 0** (mix the whole spectrum) | Does band targeting matter? (H4) |
| A2 | E5 with B chosen uniformly from all 8 clusters (no OHEM) | Does picking the dominant bands matter? (H4) |
| A3 | E5 with δ = 0 (no CAM consistency) | Value of CAM consistency |
| A4 | E5 without confidence sampling | Value of confidence sampling |
| A5 | E6 with single BN (no split) | DAT's largest ablation effect, on GTSRB |
| A6 | E6 with A_G replaced by the amplitude of a random same-class image (no generator) | Value of the learned generator |

Optional if time allows: E5 and E6 with single BN for both (fairness check); E5 using E1 instead of E0 as the OHEM model.

---

## 6. Fo-Mixup for 32×32 signs (Arm A detail)

Implement in `src/fomixup.py`. All operations float32.

### 6.1 Spectrum
For each image and channel: `F = fftshift(fft2(x_c))`, `A = |F|`, `P = angle(F)`. Centre (DC) at index (16, 16).

### 6.2 Segment partition [adaptation of FreqDebias "angular segments"]
For each frequency position (u, v) with offsets `du = u − 16`, `dv = v − 16`:
- radius `r = sqrt(du² + dv²)`; ring index `floor(r / 2)` → rings 0–11.
- angle `θ = atan2(dv, du) mod π` (folding by π keeps the mask conjugate-symmetric); sector index `floor(θ / 15°)` → sectors 0–11.
- DC position: ring 0, sector 0.
- Segment = (ring, sector): up to 144 segments. Fixed for all images; compute once.

### 6.3 Per-image dominant-band masks (precomputed once, offline) [paper: FreqDebias Alg. 1, steps 3–10]
Script: `src/precompute_fomixup.py`. Requires the trained **E0 best.pt** as the frozen "OHEM model" [paper: FreqDebias uses a pre-trained standard detector].
For every training image:
1. `μ_seg` = mean of `log(1 + A)` over the positions of each segment, averaged over the 3 channels.
2. Cluster the up-to-144 scalar `μ_seg` values into **k = 8** clusters with 1-D K-means [paper: k = 8]. Implement a batched, vectorised 1-D K-means in torch on GPU (20 Lloyd iterations, initial centres at the 8 quantiles). Verify against scikit-learn on 20 images in a test.
3. For each cluster z: binary mask `B_z` (positions of its segments, same mask for all channels). Filtered image `x_z = real(ifft2(ifftshift(A·B_z · exp(i·P))))`, clamped to [0, 1].
4. Feed all 8 filtered images to the frozen OHEM model (BN-A, eval) and compute CE against the true label.
5. Keep the **t = 3** clusters with the **highest** loss [paper: t = 3].
6. Store per image: the cluster index of each segment (uint8[144]) and the 3 chosen cluster IDs. Save to `OUT/data_cache/fomixup_masks.pt`.

### 6.4 The mix [paper: FreqDebias Eq. 1–2]
For image x_i with partner x_j (same class):
```
B   = one of x_i's 3 stored masks, chosen uniformly at random
ξ   ~ U(0, 1)
Â   = A_i ⊙ B + ((1 − ξ)·A_i + ξ·A_j) ⊙ (1 − B)
Â   = max(0, p ⊙ Â),   p ~ N(1, 0.1²) element-wise      [adaptation: paper prints N(1, 0), which is degenerate]
x̃   = clip_[0,1]( real( ifft2( ifftshift( Â · exp(i·P_i) ) ) ) )
```
**Implement Eq. 1 exactly as printed: A_i is kept inside B and mixing happens outside B.** Because B is the band whose *band-only* image gives the highest loss (the band the model relies on least), the mixed region (1 − B) is where the model's dominant components lie. Do not swap B and (1 − B).

### 6.5 Visual check (required at Milestone 3)
Save `OUT/figures/fomixup_grid.png` (16 signs: original, x̃ at ξ = 0.25/0.5/0.75, and the mask B) and `OUT/figures/dat_grid.png` (16 signs: original, x̂ at λ = 0.25/0.5/0.75). The sign's shape must stay readable while colour/contrast changes.

---

## 7. Evaluation protocol (`src/evaluate.py`)

Every model is evaluated from `best.pt`, float32, eval mode, **BN-A**, on the official test set (or its fixed subsets below).

| Metric | Exact setting | Test images |
| --- | --- | --- |
| Clean accuracy | — | all 12,630 |
| FGSM | ε = 8/255 | all |
| PGD-20 | ε = 8/255, α = 2/255, random start, CE | all |
| PGD-100 | as PGD-20, 100 steps | all |
| CW∞ | PGD-30 with CW margin loss (κ = 0), α = 2/255 | all |
| AutoAttack | `autoattack` package, version `'standard'`, norm Linf, ε = 8/255 | **fixed subset: first 2,000 indices** of the test set (indices saved to `OUT/split/aa_subset.txt`) |
| ε sweep | PGD-20 at ε ∈ {2, 4, 8, 12, 16}/255, α = ε/4 | all |
| Corruptions | 6 corruptions × 5 severities (Section 3.3) | all |
| Patch attack | per-image square patch of side 4, 6, 8 px at a random location in the central 24×24 region (location seed = image index); patch pixels free in [0, 1]; 100 steps, step 0.05, maximize CE | **first 2,000** |
| Per-class | clean and PGD-20 accuracy for all 43 classes | all |
| Cost | training sec/epoch (from log); inference latency at batch 1 (mean of 200 runs after 20 warm-ups) and throughput at batch 256, on GPU and CPU | — |

**Safety-critical classes** for the per-class table: Stop (14), Yield (13), No entry (17), speed limits (0, 1, 2, 3, 4, 5, 7, 8). **Verify these IDs against the official GTSRB class-name list** before reporting; record the list in `OUT/split/class_names.json`.

### 7.1 Gradient-masking sanity checks (automatic, reported for every model)
- PGD-100 ≤ PGD-20 + 0.5
- AutoAttack ≤ PGD-20 (on the same 2,000 images) + 0.5
- Square attack accuracy (from AutoAttack's per-attack log) ≥ the white-box AutoAttack accuracy − 0.5
- ε-sweep accuracy is non-increasing in ε

Any failure is written to `metrics.json` and printed in red. A model failing a check must not be reported as robust without investigation.

---

## 8. Result format

### 8.1 Directory layout (on Google Drive)
```
OUT/                                   # e.g. /content/drive/MyDrive/gtsrb_freq_robust
  env/requirements.lock.txt
  split/train.txt, val.txt, stats.json, aa_subset.txt, class_names.json
  data_cache/fomixup_masks.pt, gtsrb_c/
  figures/fomixup_grid.png, dat_grid.png, corruptions/
  runs/<EXP>/seed<k>/
      config.yaml
      checkpoints/last.pt, best.pt
      train_log.csv
      metrics.json
      per_class.csv
      eps_sweep.csv
      corruptions.csv
      patch.csv
  results/                             # written by aggregate.py
      table_main.md / .csv
      table_corruption.md / .csv
      table_safety_classes.md / .csv
      table_cost.md / .csv
      table_ablation.md / .csv
      fig_eps_curve.png
      fig_corruption_severity.png
      fig_train_curves.png
```
`<EXP>` ∈ {E0_standard, E1_pgdat, E2_trades, E3_base, E4_fd_clean, E5_fd_at, E6_dat, A1 … A6}.

### 8.2 `train_log.csv` columns
`epoch, lr, train_loss, train_clean_acc, val_clean_acc, val_pgd20_acc, sec_per_epoch, is_best`
(val_pgd20_acc empty on epochs where it is not computed). For E5/E6 add the separate loss terms: `l_at_x, l_at_xtilde, l_consistency, l_att` (E5) or `l_at_x, l_at_xhat, l_js, gen_loss` (E6).

### 8.3 `metrics.json` schema (accuracies in %, two decimals)
```json
{
  "exp": "E6_dat", "seed": 0,
  "git_commit": "...", "torch": "...", "torchvision": "...", "python": "...", "gpu": "Tesla T4",
  "epochs": 60, "best_epoch": 47, "train_hours": 0.0, "sec_per_epoch": 0.0,
  "clean": 0.0, "fgsm": 0.0, "pgd20": 0.0, "pgd100": 0.0, "cw_inf": 0.0,
  "autoattack": 0.0, "autoattack_n": 2000, "autoattack_per_attack": {"apgd-ce": 0.0, "apgd-t": 0.0, "fab-t": 0.0, "square": 0.0},
  "eps_sweep": {"2": 0.0, "4": 0.0, "8": 0.0, "12": 0.0, "16": 0.0},
  "corruption_mean": 0.0,
  "corruptions": {"gaussian_noise": [0, 0, 0, 0, 0], "gaussian_blur": [], "brightness": [], "contrast": [], "fog": [], "jpeg": []},
  "patch": {"4": 0.0, "6": 0.0, "8": 0.0, "n": 2000},
  "safety_classes": {"14": {"name": "Stop", "clean": 0.0, "pgd20": 0.0}},
  "latency_ms_bs1_gpu": 0.0, "throughput_bs256_gpu": 0.0, "latency_ms_bs1_cpu": 0.0,
  "sanity": {"pgd100_le_pgd20": true, "aa_le_pgd20": true, "square_not_stronger": true, "eps_monotonic": true}
}
```

### 8.4 Final tables (`aggregate.py`; mean ± std over available seeds; seed count shown)
- **table_main:** Method | Seeds | Clean | FGSM | PGD-20 | PGD-100 | CW∞ | AutoAttack | Patch-6px
- **table_corruption:** Method | Clean | mean over all corruptions | per-corruption mean over severities
- **table_safety_classes:** Method | per safety class: clean / PGD-20
- **table_cost:** Method | sec/epoch | train hours | GPU latency (ms) | CPU latency (ms)
- **table_ablation:** Variant | Clean | PGD-20 | AutoAttack | Corruption mean, with the difference from its parent (E5 or E6)

Rows in `table_main` in this order: E0, E1, E2, E3, E4, E5, E6. Bold the best value per column in the Markdown version.

### 8.5 Figures
- `fig_eps_curve.png`: PGD-20 accuracy vs ε, one line per method (mean, shaded ± std).
- `fig_corruption_severity.png`: accuracy vs severity, averaged over corruptions, one line per method.
- `fig_train_curves.png`: validation clean and PGD-20 vs epoch for E3, E5, E6 (seed 0).

---

## 9. Run plan on Colab (priority order)

Free Colab cannot be relied on for all ~25 runs, so run in tiers. Each tier is a complete, reportable result.

| Tier | Runs | Gives |
| --- | --- | --- |
| 1 (minimum thesis result) | E0, E3, E5, E6 — seed 0 | The core comparison |
| 2 | E1, E2, E4 — seed 0 | Standard baselines and H2 |
| 3 | Seeds 1, 2 of E3, E5, E6 | Error bars on the core comparison |
| 4 | A1–A6 | Why the results differ |
| 5 | Seeds 1, 2 of E1, E2 | Error bars on baselines |

Dependency: **E0 must finish before** `precompute_fomixup.py`, which must finish before E4, E5, A1–A4.

Run `evaluate.py` immediately after each training run finishes, so results accumulate even if later tiers are never reached.

---

## 10. Colab procedure (`notebooks/colab_runner.ipynb`)

### 10.1 One-time setup
1. Runtime → Change runtime type → **GPU**.
2. Cell: `!nvidia-smi` — print a clear warning and stop if no GPU.
3. Cell: mount Drive — `from google.colab import drive; drive.mount('/content/drive')`.
4. Cell: set `OUT = '/content/drive/MyDrive/gtsrb_freq_robust'`, `DATA = '/content/data'`, `REPO = '/content/repo'`.
5. Cell: clone or pull the repo (use `GITHUB_TOKEN` from Colab Secrets if private); `!pip install -q -r requirements.txt`.
6. Cell: download GTSRB into `DATA` (skips if already present this session).
7. Cell (first time only): `!python -m src.make_split --data $DATA --out $OUT`.

### 10.2 One cell per run, for example
```
!python -m src.train    --config configs/E6_dat.yaml --seed 0 --data $DATA --out $OUT --resume
!python -m src.evaluate --exp E6_dat --seed 0 --data $DATA --out $OUT
```
Plus:
```
!python -m src.precompute_fomixup --data $DATA --out $OUT        # after E0 only
!python -m src.aggregate --out $OUT
```

### 10.3 Resuming after a disconnect
- `train.py` writes `last.pt` (model, both BN sets, optimizer(s), generator + its optimizer for DAT, scheduler, GradScaler, epoch, RNG states, best metric) **every epoch**, atomically (write to a temp file, then rename).
- After a disconnect: reconnect, re-run cells 10.1 steps 2–6, then re-run the **same** training cell. `--resume` continues from `last.pt`.
- `evaluate.py` is also resumable: it skips metrics already present in `metrics.json` (so a long AutoAttack run is not repeated).
- Progress check without stopping a run: open `runs/<EXP>/seed<k>/train_log.csv` on Drive.

---

## 11. Repository layout and tests

```
.
├── CLAUDE.md
├── README.md            # local + Colab instructions, credits (DAT code, papers)
├── NOTES.md             # every adaptation or paper/code discrepancy
├── requirements.txt
├── configs/             # E0_standard.yaml … E6_dat.yaml, A1.yaml … A6.yaml
├── notebooks/colab_runner.ipynb
├── src/
│   ├── data.py  make_split.py  corruptions.py
│   ├── models.py        # CIFAR ResNet-18, normalization layer, dual BN, CAM access
│   ├── attacks.py       # FGSM, PGD (CE / CW margin / KL), EAEG, patch
│   ├── fourier.py       # rfft2/fft2 helpers, float32
│   ├── fomixup.py  precompute_fomixup.py
│   ├── aag.py           # DAT generator
│   ├── methods/         # standard, pgd_at, trades, base_at, fd (E4/E5), dat
│   ├── train.py  evaluate.py  aggregate.py
└── tests/
```

Every script supports `--smoke` (CPU, 1 epoch, 2 batches, < 1 minute).

**Required tests (all must pass before any GPU run):**
- Track split: no track ID in both train and val.
- No flip transform anywhere in training transforms.
- FFT round trip reproduces x within 1e-5 (both rfft2 and fftshift(fft2) paths).
- Fo-Mixup output is real-valued (max |imag| < 1e-4 before taking `.real`), in [0, 1], and equals x when ξ = 0 and p ≡ 1.
- Segment masks are conjugate-symmetric.
- 1-D K-means matches scikit-learn cluster assignments on 20 images (up to label permutation).
- PGD/EAEG outputs stay within ε (L∞) of the input and inside [0, 1].
- Dual BN: branches A and B use different parameters; test-time forward uses A.
- Resume: train 2 epochs → stop → resume 1 epoch equals 3 straight epochs (deterministic mode, same seed).
- `--smoke` end-to-end for every experiment config.

---

## 12. Milestones — stop after each and report

At each stop, report: what was built, test results, any deviation from this file (and the `NOTES.md` entry), and the exact Colab cells to run next.

- **M1 — Pipeline + E0.** Data, split, model, attacks, training/eval scripts, notebook, all tests. *Check:* E0 clean test accuracy well above 95%; E0 PGD-20 accuracy near 0%.
- **M2 — E3 (then E1, E2).** *Check:* robust accuracy clearly above 0%, sanity checks pass. **Report sec/epoch on the Colab GPU** and estimated hours per run, so the student can plan tiers. Decide here whether 60 epochs is enough (Section 4.3).
- **M3 — Fo-Mixup.** Precompute masks from E0, produce `fomixup_grid.png`, then E5 (seed 0) and E4. *Check:* the grid shows readable signs; no NaNs.
- **M4 — DAT.** Port from the official repo, produce `dat_grid.png`, run E6 (seed 0). *Check:* stable training, sanity checks pass.
- **M5 — Tier 1 complete:** full evaluation + `aggregate.py` tables for E0, E3, E5, E6.
- **M6 — Remaining tiers** as the student schedules them.

---

## 13. Rules

- Never change splits, ε, α, K, β, epochs or the evaluation subsets for one method only.
- Never use the test set for training or model selection.
- Always report `best.pt` (best validation PGD-20), never the last epoch.
- Do not invent DAT generator details — read them from the official code.
- Implement Fo-Mixup's Eq. 1 exactly as in Section 6.4; do not swap B and (1 − B).
- Do not describe Arm A or any ablation as novel; the student will check the literature.
- Comment every formula in the code with its equation number from this file or the paper, so the student can explain it in a defense.

---

## 14. Extension: colour-aware frequency mixing (student extension, added 2026-11-28)

**Motivation.** E5 and E6 compute the FFT of each RGB channel and perturb each channel's amplitude independently. Colour is carried by the relation between R, G and B, so this also shifts colour; for traffic signs colour carries meaning (red = prohibition/warning, blue = mandatory). The variants below apply the same amplitude perturbation in YCbCr: Y (luminance) is perturbed as E5/E6 perturb each channel, Cb/Cr (chrominance) are kept. Do not describe this as novel (§13); the student checks the literature.

### 14.1 Phase 0 — diagnosis (no training; decide before Phase 2)
- Colour groups (verified against the official class names, `src/data.py::COLOUR_GROUPS`): red 0–5, 7–11, 13–31; blue 33–40; yellow 12; white 6, 32, 41, 42.
- `src/colour_groups.py`: per-group clean and PGD-20 accuracy from the per-image results of `best.pt` (`eval_flags.pt`), and for E5/E6 the difference from E3 per group with a 95 % paired bootstrap interval (2,000 resamples, images resampled jointly), plus the colour gap = drop on red+blue − drop on white.
- Proceed to Phases 1–3 only if E5 and/or E6 show a colour gap whose interval excludes 0 (colour-coded signs lose more). Otherwise reconsider.

### 14.2 Phase 1 — method
- `src/color.py`: RGB ↔ YCbCr (ITU-R BT.601 full range), differentiable, float32 (like all Fourier arithmetic, §4.3).
- **E5-CA** (`fomixup_color`): Fo-Mixup Eq. 1–2 applied per YCbCr channel with strength w = (1, 1 − α, 1 − α) on ξ and on the Eq. 2 noise; channels with w = 0 pass through unchanged. Same band masks B as E5 (selected once from the RGB spectrum by §6.3).
- **E6-CA**: the DAT mix applied in YCbCr. With α = 1 the generator outputs one channel (the Y amplitude) and only Y is mixed; with α < 1 it outputs three channels and λ is scaled per channel by w. Everything else (conditioning, generator optimiser, EAEG on x̂, JS consistency) as in §5.7.
- **α = `chroma_protect`** ∈ [0, 1]: 1 = colour untouched (E5-CA/E6-CA), 0 = all three YCbCr channels perturbed. The per-RGB-channel original is E5/E6 itself (`color_space: rgb`); α = 0 in YCbCr is close to, but not identical with, RGB perturbation because amplitude mixing is non-linear (NOTES.md N23).

### 14.3 Phase 2 — runs (seed 0)
| ID | Config | Differs from parent only by |
| --- | --- | --- |
| E5-CA | `configs/E5_fd_at_ca.yaml` | `color_space: ycbcr`, `chroma_protect: 1.0` (parent E5) |
| E6-CA | `configs/E6_dat_ca.yaml` | `color_space: ycbcr`, `chroma_protect: 1.0` (parent E6) |

Same base, schedule, BN split, masks and evaluation as the parents (§13 fairness). An α sweep (e.g. 0, 0.5) is optional: copy a config and change `chroma_protect`.

### 14.4 Phase 3 — evaluation
- Full §7 evaluation of E5-CA and E6-CA.
- `table_colour_aware`: overall metrics and the difference from the parent.
- `table_colour_groups` / `table_colour_drop`: per colour group, and E5-CA vs E5, E6-CA vs E6 with paired bootstrap intervals.
- Success condition: higher accuracy on the colour-coded groups (interval excludes 0) without losing overall robustness (AutoAttack, PGD-20 within noise of the parent). One seed: differences under ~1 point are noise.
