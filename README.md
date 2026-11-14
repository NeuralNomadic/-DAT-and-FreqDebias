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
notebooks/    colab_runner.ipynb
```

## Colab (recommended)
Open `notebooks/colab_runner.ipynb` in Google Colab with a GPU runtime, set `GITHUB_REPO` in cell 5 (and a
`GITHUB_TOKEN` Colab Secret if the repo is private), run the setup cells, then one cell per run. Everything
is written to `OUT = /content/drive/MyDrive/gtsrb_freq_robust` and is resumable (`--resume`).

Dependency order: `E0_standard` → `src.precompute_fomixup` → E4 / E5 / A1–A4.

## Local
```bash
pip install -r requirements.txt          # torch/torchvision: use what is installed (Colab: preinstalled)
python -m pytest tests -q                # all tests must pass before any GPU run (CPU, ~5 min)
python -m src.make_split --data DATA --out OUT
python -m src.train    --config configs/E0_standard.yaml --seed 0 --data DATA --out OUT --resume
python -m src.evaluate --exp E0_standard --seed 0 --data DATA --out OUT
python -m src.precompute_fomixup --data DATA --out OUT
python -m src.aggregate --out OUT
```
Every script accepts `--smoke` (CPU, 1 epoch, 2 batches, synthetic data if GTSRB is absent).

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
  (`autoattack` package). Corruptions: Hendrycks & Dietterich, ICLR 2019 (`imagecorruptions` package).
* Dataset: GTSRB, Stallkamp et al., 2012 (via `torchvision.datasets.GTSRB`).
