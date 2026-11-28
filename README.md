# Frequency-domain adversarial robustness for traffic-sign recognition (GTSRB)

FreqDebias-style dominant-band amplitude mixing (Arm A) vs DAT generative amplitude mixing (Arm B), both on
the same EAEG adversarial-training base, ResNet-18 (CIFAR variant), 32×32 GTSRB. The study design is in
`CLAUDE.md`; every deviation from it is in `NOTES.md`.

## Layout
```
configs/      _defaults.yaml (shared, never per-method) + E0…E6, A1…A6
src/          data, make_split, corruptions, models, attacks, fourier, fomixup, precompute_fomixup, aag,
              methods/ (standard, pgd_at, trades, base_at, fd, dat), train, evaluate, aggregate, make_grids
tests/        required tests (CLAUDE.md §11)
notebooks/    colab_runner_v2.ipynb   (use this one; colab_runner.ipynb is the old v1, kept for reference)
```

## Colab (recommended)
1. Colab → File → Open notebook → GitHub → `NeuralNomadic/-DAT-and-FreqDebias` → `notebooks/colab_runner_v2.ipynb`.
2. Runtime → Change runtime type → T4 GPU.
3. Optional but recommended: a free Hugging Face account and a **Write** access token stored as the Colab secret
   `HF_TOKEN`. Results then go to a private Hugging Face dataset; without it the notebook falls back to Google
   Drive automatically.
4. Run cells 1–6 (settings, GPU, code, storage, data, helpers), then one run cell at a time.

Training writes to the local VM disk (`/content/out`); `src/storage.py` syncs results to storage every
10 minutes and after every step. After a disconnect, run cells 1–6 again and re-run the step you were on: it
resumes from the last checkpoint (`--resume`).

Dependency order: `E0_standard` → `src.precompute_fomixup` → E4 / E5 / A1–A4.

## Extension: colour-aware frequency mixing
`CLAUDE.md` §14 and the last cells of the notebook: Phase 0 diagnosis (`python -m src.colour_groups --out OUT`,
no GPU), then E5-CA / E6-CA (`configs/E5_fd_at_ca.yaml`, `configs/E6_dat_ca.yaml`), which mix only the luminance
(Y) spectrum in YCbCr and keep the sign's colour. Details and every choice made: `NOTES.md` N23.

## Local
```bash
pip install -r requirements.txt          # torch/torchvision: use what is installed (Colab: preinstalled)
python -m pytest tests -q                # all tests must pass before any GPU run (CPU, ~12 min)
python -m src.make_split --data DATA --out OUT
python -m src.train    --config configs/E0_standard.yaml --seed 0 --data DATA --out OUT --resume
python -m src.evaluate --exp E0_standard --seed 0 --data DATA --out OUT
python -m src.precompute_fomixup --data DATA --out OUT
python -m src.aggregate --out OUT
```
Every script accepts `--smoke` (CPU, 1 epoch, 2 batches, synthetic data if GTSRB is absent).
Optional sync outside Colab: set `OUT_SYNC=hf:<user>/<repo>` (with `HF_TOKEN`) or `OUT_SYNC=dir:<folder>`.

## Results
`OUT/runs/<EXP>/seed<k>/` holds `config.yaml`, `train_log.csv`, `checkpoints/{last,best}.pt`, `metrics.json`,
`per_class.csv`, `eps_sweep` (in metrics), `corruptions.csv`, `patch.csv`. `OUT/results/` holds the thesis
tables (`table_main`, `table_corruption`, `table_safety_classes`, `table_cost`, `table_ablation`, .md and .csv)
and figures (`fig_eps_curve.png`, `fig_corruption_severity.png`, `fig_train_curves.png`).

## Credits
* **DAT** — Li, Li, Wu, Tian, Zhou. *DAT: Improving Adversarial Robustness via Generative Amplitude Mix-up in
  Frequency Domain*, NeurIPS 2024. The generator (`src/aag.py`), EAEG attack, loss and training loop details
  are ported from the official code https://github.com/Feng-peng-Li/DAT (MIT License, © 2024 Fengpeng Li).
* **FreqDebias** — Kashiani et al. *Freqdebias: Towards generalizable deepfake detection via consistency-driven
  frequency debiasing*, CVPR 2025. Fo-Mixup (`src/fomixup.py`) is implemented from the paper as specified in
  `CLAUDE.md` §6 (no official code).
* PGD-AT: Madry et al., ICLR 2018. TRADES: Zhang et al., ICML 2019. AutoAttack: Croce & Hein, ICML 2020
  (`autoattack` package). Corruptions: Hendrycks & Dietterich, ICLR 2019 (formulas and constants of the
  `imagecorruptions` package, reimplemented in `src/corruptions.py`; NOTES.md N13).
* Dataset: GTSRB, Stallkamp et al., 2012 — official final training and test archives from the GTSRB server
  (NOTES.md N18).
