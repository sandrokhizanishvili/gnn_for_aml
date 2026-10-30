# Experiments run from this repository

Status and pointers only, no numbers. The finished Kaggle runs of the thesis stay in `../gnn_for_fraudulent_patterns`
(`EXPERIMENTS.md` there); this file tracks what is run with this pipeline. ✅ done · 🔄 running · ⬜ todo

Two settings differ from the Kaggle runs of the sister repository, for every run made here:
- the best epoch is chosen on validation PR-AUC for every model (the Kaggle grid used validation F1; the thesis
  tuning protocol already used PR-AUC); LightGBM's kept round is the best validation average precision instead of
  the lowest weighted validation loss;
- every final retrain and the tabular cross-check use five seeds (42–46; the sister repository's tabular runs used four).

## GNN search — every operator × configs 2 and 4 (Cineca)

Random search over the space of `conf/hyperparameters/gnn/search.yaml` (lr, hidden, norm, residual, dropout,
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

Commands: README section 3. Hand-off: Loris clones the repo on Leonardo, creates the environment, runs `mode=check`,
`tuning=gnn/search` (re-run until `aggregate.py` reports every seed done), `aggregate.py`, `tuning=gnn/final`,
`aggregate.py`, then sends `results_small.tar.gz` and `results_predictions.tar.gz`.

## Fixed-architecture runs (re-runs with seeds, same code)

The grid (4 × 7), the node-encoding study (configs 4 and 5, rwpe16) and the tuned-GFP study (configs 4 and 5; needs
`rwpe/` and `gfp_variants/` next to the graphs): one command line each, README section 3. ⬜ not started; the
Kaggle results of the sister repository remain the thesis numbers until a seed sweep is decided.

## Cross-checks without message passing (same code)

`model=lightgbm,mlp seed=42..46` at the frozen thesis winners; `tuning=lightgbm/search` / `tuning=mlp/search` (one
job each) then `tuning=<model>/final`. ⬜ not started; the Kaggle / local results of the sister repository
(`Outputs/GBT/`, `Outputs/MLP/`) remain the thesis numbers. The tail-window appendix check of the sister repository
is not ported (needs the raw CSV and the WSL GFP stream; outside the protocol).

## Data preparation

`prepare_data.py` rebuilds every input from the two raw IBM files in `data/raw/` (README section 1). Reference
output: the sister repository's `Data/` (copied into `data/`). Verification status: ✅ full rebuild on 30 Oct 2026
(`data_rebuilt/`, 3 steps, about 40 minutes with the GFP stream in WSL) compared file by file:

| File | Result |
|---|---|
| `rwpe/` (6 files), `gfp_variants/tuned.npy`, `tuned_cols.json` | identical |
| graphs: node features, edges, times, labels, masks, the 20 baseline columns, 41 of the 61 GFP columns | identical |
| graphs: the 20 GFP amount statistics (avg / sum / var / skew / kurtosis of amounts per account) | cells differing by more than 1e-3: 0.03 %; by more than 0.1: 0.0013 % (near-constant accounts, where skew / kurtosis are numerically undefined) |
| `feature_meta.json` | same columns and sizes; the old file carries extra notes, the new one `split_rows` |

Cause: snapml is deterministic (checked: repeated runs, 8 threads and 1 thread, identical), so the difference is in
its input, the float64 USD amounts. The original exchange rates were downloaded in September 2022 and are recoverable
from the stored features only to float32 precision (`Amount_Log` is identical), so a few amounts differ below that
resolution. Sums then differ in the last bit and the unstable statistics of near-constant accounts flip. This is
the limit of what a rebuild can reproduce; the thesis files in `data/` stay the data every run uses.

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
- 2026-10-30 — full data rebuild by Sandro (`data_rebuilt/`) compared with the thesis files: see "Data preparation".
