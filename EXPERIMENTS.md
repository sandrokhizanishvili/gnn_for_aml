# Experiments run from this repository

Status and pointers only, no numbers. The finished Kaggle runs of the thesis stay in `../gnn_for_fraudulent_patterns`
(`EXPERIMENTS.md` there); this file tracks what is run with this pipeline. ✅ done · 🔄 running · ⬜ todo

Two settings differ from the Kaggle runs of the sister repository, for every run made here:
- the best epoch is chosen on validation PR-AUC for every model (the Kaggle grid used validation F1; the thesis
  tuning protocol already used PR-AUC); LightGBM's kept round is the best validation average precision instead of
  the lowest weighted validation loss;
- every final retrain and the tabular cross-check use five seeds (42–46; the sister repository's tabular runs used four).

## GNN search — every operator × configs 2 and 4 (Cineca)

Random search over the space of `conf/hyperparameters/gnn/space.yaml` (lr, hidden, norm, residual, dropout,
weight decay, pos_weight, readout layers and width; depth 1–10 with the fixed neighbour table), 100 trials × 2 seeds
(42, 43) per pair, 100 epochs at a constant LR, no early stopping, best epoch = argmax of the seed-mean validation
PR-AUC, winner = highest mean validation PR-AUC (tie-break validation F1). Then a 5-seed retrain (42–46) on train + val
at the frozen epoch, one threshold = the mean of the two tuning thresholds, test scored once per seed.
The protocol follows Luo et al. (2024).

Differences from the plan of the sister repository (`EXPERIMENTS.md` §2b there), decided with the rewrite:
- no memory probe: every depth 1–10 is in the draw; a trial that does not fit the GPU is recorded `oom` and cannot win;
- no per-epoch checkpoints during tuning (the trial's metrics come from `history.csv`; the retrain starts from scratch);
- loader workers fixed (4), job memory fixed (64 GB); the trial list is drawn from the same seed but, without the
  probe, maps to different cells than the old `trials_space.json`.

| Pair | Search (`outputs/tuning/<model>/<run>/`) | Final (`outputs/final/<model>/<run>/seed-<s>/`) |
|---|---|---|
| GIN c2 (base / base) | ⬜ | ⬜ |
| GIN c4 (full / full) | ⬜ | ⬜ |
| PNA c2 | ⬜ | ⬜ |
| PNA c4 | ⬜ | ⬜ |
| GATv2 c2 | ⬜ | ⬜ |
| GATv2 c4 | ⬜ | ⬜ |
| Transformer c2 | ⬜ | ⬜ |
| Transformer c4 | ⬜ | ⬜ |

Commands: README, "Experiments". Hand-off: Loris clones the repo on Leonardo, creates the environment, runs `mode=check`,
`tuning=gnn/search` (re-run until `aggregate.py` reports every seed done), `aggregate.py`, `tuning=gnn/final`,
`aggregate.py`, then sends `results_small.tar.gz` and `results_predictions.tar.gz`.

## Fixed-architecture runs (re-runs with seeds, same code)

The grid (4 × 7), the node-encoding study (configs 4 and 5, rwpe16) and the tuned-GFP study (configs 4 and 5; needs
`rwpe/` and `gfp_variants/` next to the graphs): one command line each (README, "Experiments"). ⬜ not started; the
Kaggle results of the sister repository remain the thesis numbers until a seed sweep is decided.

## Cross-checks without message passing (same code)

`tuning=lightgbm/search` / `tuning=mlp/search` (one job each) then `tuning=<model>/final` (the winner of
`recipe.json` x seeds 42–46). The searches are redone here rather than importing the sister repository's winners
(LightGBM trial 84, MLP trial 28): every fit now stops on validation PR-AUC, so the trial scores and possibly the
winner change, and one chain search → final from this repo leaves no second "best" to explain. ⬜ not started; until
then the Kaggle / local results of the sister repository (`Outputs/GBT/`, `Outputs/MLP/`) remain the thesis numbers.
The tail-window appendix check of the sister repository is not ported (needs the raw CSV and the WSL GFP stream;
outside the protocol).

## Data preparation

`prepare_data.py` rebuilds every input from the two raw IBM files in `data/raw/` (README, "Quick start"). Reference
output: the sister repository's `Data/` (copied into `data/`). Verification status: ✅ full rebuild on 30 Oct 2026
(3 steps, about 40 minutes with the GFP stream in WSL) compared file by file: `train/val/test_graph.pt` (every
attribute), the six `rwpe/` files, `gfp_variants/tuned.npy` and `tuned_cols.json` are **bit-identical** to the thesis
files; `feature_meta.json` has the same columns and sizes (the old file carries extra notes, the new one `split_rows`).

One fix was needed on the way: the first rebuild differed in the 20 GFP amount statistics (0.03 % of cells by more
than 1e-3). snapml is deterministic (repeated runs, 8 and 1 threads: identical), so the cause was its input, the USD
amounts: pandas' default CSV float parser read 80 of the 140 frozen rates one bit off. `prepare_data.py` now reads
`conf/fx_rates.csv` with `float_precision='round_trip'`; the rates then equal the Yahoo values bit for bit and the
rebuilt graphs equal the thesis graphs.

## Log

- 2026-10-27 — repository created as `aml_hydra`: `run.py`, `aggregate.py`, `aml/` (data, model, train, tuning),
  `conf/`; verified on a 40k-edge slice: every mode, the sweep path (tune → aggregate → final → aggregate, reruns skip
  finished work) with the in-process launcher, the Leonardo launcher config composed, and the old-vs-new regression
  against `gnn_core.py` (every operator on configs 2 and 4, v1 and a Luo-style recipe).
- 2026-10-30 — `aml/tabular.py` (LightGBM + MLP) and its recipes; verified on the slice against the old
  `gbt_core.py` / `mlp_core.py` (identical search draws, kept round / best epoch, threshold and predictions).
- 2026-10-30 — restructured and renamed to `gnn_for_aml`: `pipeline/` with one folder per model, config groups
  `hyperparameters/<model>/{fixed,search}`, `feature_config`, `machine`, `tuning/<model>/{search,final}`; outputs
  under `outputs/train | tuning | final/<model>/`; own `data/` (raw files in `data/raw/`). Verified byte-identical
  against `aml_hydra` on the slice: GNN train (gin, pna × c2, c4), search (2 trials × 2 seeds) → aggregate → final →
  aggregate, LightGBM and MLP train → search → final, the 100-trial draw, the multirun path, the Leonardo config.
  Then the best-epoch rule was switched to validation PR-AUC for every model (see the top of this file) and the
  slice runs repeated. Direct check against the pre-refactor thesis `gnn_core.py` (same process, one thread): every
  operator on configs 2 and 4 at the fixed recipe plus two tuned recipes, identical history, metrics and weights.
- 2026-10-30 — full data rebuild compared with the thesis files: bit-identical after the exchange-rate parser fix
  (see "Data preparation").
- 2026-10-30 — LightGBM port checked on the full data against the sister repository's seed-42 run (trial 84, local
  CPU, 16 threads): the 59 common `history.csv` rows (rounds 25–1475) identical to the last digit, the first 1,380
  trees of `best.txt` identical text. Only the kept round differs, by the new rule: PR-AUC still rising at the cap,
  round 1,993 kept (old: lowest weighted validation loss, round 1,380); test F1 0.517 vs 0.525, test PR-AUC 0.484
  vs 0.482. Check outputs deleted, not a result. Then the tabular `fixed.yaml` files were removed: a searched winner
  lives only in `recipe.json`, LightGBM and the MLP run as search → final only (`README.md`, `CLAUDE.md`);
  `hyperparameters/<model>/search.yaml` renamed `space.yaml`. A readability pass over the Python (imports, the
  tabular input plumbing, three docstrings) was checked on the 40k slice, old tree vs new: every mode of every model
  (check, GNN train / tune / aggregate / final, LightGBM and MLP search / final), 34 output files identical.
