# Graph neural networks for anti-money-laundering detection

Master thesis, Sapienza University of Rome.

Do engineered graph features help graph neural networks find money-laundering patterns in transaction networks?
Edge classification on the IBM AML HI-Small dataset (Altman et al., NeurIPS 2023), cumulative temporal snapshots,
60/20/20 temporal split. Four operators (GIN, PNA, GATv2, graph Transformer) share one model template; the
variables are which edge features enter message passing and the readout (`c1..c7`), a node encoding (RWPE), the
GFP feature sheet and the hyperparameters. LightGBM and an MLP on the same inputs are the cross-check without
message passing. One config = one run (`python run.py` + overrides); `-m` runs one job per combination of the
listed values. Configuration by Hydra, everything else plain Python.

## Repository structure

```
prepare_data.py          data/raw (two IBM files) -> data/: graphs, RWPE encodings, the tuned GFP sheet
run.py                   one run: mode train | tune | final | check, any model
aggregate.py             after a GNN search: scores the trials, freezes each winner into recipe.json
conf/
  config.yaml            one run: mode, seed, node_encoding, gfp_sheet; defaults of every group below
  prepare_data.yaml      data preparation settings; fx_rates.csv = the frozen USD exchange rates
  model/                 gin pna gat transformer lightgbm mlp
  hyperparameters/       gnn/fixed.yaml = the fixed architecture (mode train); gnn/space.yaml = fixed + what the
                         search changes + the search space. lightgbm/, mlp/: space.yaml only (the never-searched
                         settings + the search space); their winner lives in recipe.json, never in a yaml
  feature_config/        c1 .. c7: edge columns in message passing (mp) and at the readout
  machine/               local kaggle leonardo: data and output paths; leonardo adds the SLURM launcher
  tuning/                <model>/search.yaml, <model>/final.yaml: the saved search / final-retrain commands
pipeline/
  data.py                snapshots, column sets, node encodings, the train+val graph, the 93-column matrix
  gfp.py                 GFP normalisation; the snapml stream used by prepare_data.py
  metrics.py             threshold sweep, the seven metrics, per-epoch row, threshold sensitivity
  results.py             run names and folders, results.json, predictions.csv, curves.png, trial logs
  gnn/                   model.py (template; only the conv differs), train.py (loaders, epoch loop, resume), tune.py (random search)
  lightgbm/              model.py, train.py, tune.py (successive halving)
  mlp/                   model.py, train.py, tune.py (random search)
notebooks/
  gnn_fixed_architecture_runs.ipynb   Kaggle GPU notebook: one operator x c1..c7
environment.yml          torch 2.8 + CUDA 12.6, PyG 2.8.1, LightGBM, Hydra (+ SLURM launcher)
EXPERIMENTS.md           status of every experiment run from this repository
data/      (not versioned)  raw/, train|val|test_graph.pt, feature_meta.json, rwpe/, gfp_variants/
outputs/   (not versioned)  train/, tuning/, final/, logs/   (section "Outputs")
```

## Quick start

1. Environment
   ```bash
   git clone https://github.com/sandrokhizanishvili/gnn_for_aml.git && cd gnn_for_aml
   conda env create -p <env path> -f environment.yml
   ```
2. Data: put `HI-Small_Trans.csv` and `HI-Small_accounts.csv` (Kaggle, "IBM Transactions for Anti Money
   Laundering") in `data/raw/`, then
   ```bash
   python prepare_data.py                 # graphs -> rwpe -> gfp_tuned; a finished step is skipped
   python prepare_data.py step=rwpe       # one step
   ```
   The GFP step needs IBM `snapml`. Linux: `pip install snapml`, set `gfp.python=''`. Windows: the step runs in
   WSL (`python3 -m venv ~/gfp_env && ~/gfp_env/bin/pip install 'numpy<2' snapml` once). The rebuild reproduces the
   thesis data bit for bit (`EXPERIMENTS.md`, "Data preparation"). The prepared files are also the Kaggle datasets
   `sandrokhizanishvili/hi-small-gnn`, `hi-small-rwpe`, `hi-small-gfp-tuned`.
3. Check, then one run
   ```bash
   python run.py mode=check                          # versions, device, one forward/backward pass per operator
   python run.py model=gin feature_config=c5         # one fixed-architecture run
   ```
4. Result: `outputs/train/gin/gin_mp-base_readout-full_dir-in/seed-42/results.json` (section "Outputs").

## Experiments

One line each; a single run is the same line without the lists. Every run is idempotent and resumable: a finished
run returns at once, an interrupted one continues from `resume.pt`. Re-run the same command until everything is done.

| Experiment | Command | Output |
|---|---|---|
| Fixed-architecture grid (28 runs) | `python run.py -m model=gin,pna,gat,transformer feature_config=c1,c2,c3,c4,c5,c6,c7` | `outputs/train/<model>/` |
| Seed sweep of the grid | add `seed=42,43,44,45,46` | `outputs/train/<model>/<run>/seed-<s>/` |
| Node-encoding study (8 runs) | `python run.py -m model=gin,pna,gat,transformer feature_config=c4,c5 node_encoding=rwpe16` | `outputs/train/<model>/*_enc-rwpe16/` |
| Tuned-GFP study (8 runs) | `python run.py -m model=gin,pna,gat,transformer feature_config=c4,c5 gfp_sheet=tuned` | `outputs/train/<model>/*_gfp-tuned/` |
| GNN random search (1,600 jobs) | `python run.py -m tuning=gnn/search machine=leonardo` | `outputs/tuning/<model>/<run>/t<id>/` |
| Freeze the winners | `python aggregate.py machine=leonardo` | `outputs/tuning/<model>/<run>/recipe.json` |
| Final retrain of the winners (40 jobs) | `python run.py -m tuning=gnn/final machine=leonardo` | `outputs/final/<model>/` |
| Cross-check without message passing: LightGBM / MLP search (one job), then its winner x 5 seeds | `python run.py -m tuning=lightgbm/search`, then `python run.py -m tuning=lightgbm/final` (same with `mlp`) | `outputs/tuning/lightgbm/`, `outputs/final/lightgbm/` |

## Where to run

**Laptop** (`machine=local`, default): `data/` and `outputs/` inside the repository; GPU if available, else CPU.
CPU suffices for LightGBM, the MLP and `mode=check`; a GNN run needs a GPU. Multiruns run one job after another.

**Kaggle** (GNN runs): open `notebooks/gnn_fixed_architecture_runs.ipynb` in a GPU notebook with the dataset
`sandrokhizanishvili/hi-small-gnn` as input, set `MODEL`, Run All. Any command above works there with
`machine=kaggle`. The node-encoding and tuned-GFP studies also need the datasets `hi-small-rwpe` and
`hi-small-gfp-tuned` linked as `rwpe/` and `gfp_variants/` next to the graphs (`paths.data=<that folder>`).
Download `/kaggle/working/outputs/` into `outputs/` locally.

**Cineca** (Leonardo Booster, SLURM): add `machine=leonardo` on a login node; jobs are submitted by the launcher
(partition `boost_usr_prod`, one GPU per task, 24 h, requeued and resumed after the limit).

| Setting | Value |
|---|---|
| Data, environment, outputs | `$WORK/aml_thesis/data`, `$WORK/aml_thesis/env`, `$WORK/aml_thesis/outputs` |
| Environment | `conda env create -p $WORK/aml_thesis/env -f environment.yml` |
| Project account | `export ACCOUNT=<account>` in `~/.bashrc` (`saldo -b` lists it) |
| Parallel tasks | `hydra.launcher.array_parallelism=<n>` if the per-user GPU limit is smaller |
| Job logs | `outputs/logs/<timestamp>/jobs/` |

Results back from Cineca (from `$WORK/aml_thesis`):
```bash
tar -czf results_small.tar.gz --exclude='*.pt' --exclude='predictions.csv' --exclude='logs' outputs
tar -czf results_predictions.tar.gz $(find outputs/final -name predictions.csv)
```

## Outputs

```
outputs/train/<model>/<run>/seed-<s>/         results.json, history.csv, curves.png, predictions.csv, best.pt, config.yaml
outputs/tuning/<model>/<run>/t<id>/seed-<s>/  done.json, history.csv              (GNN trials: no test loader, no checkpoints)
outputs/tuning/<model>/<run>/                 t<id>/trial.json, recipe.json, trials_summary.csv, tuning_curves.png
                                              LightGBM, MLP: configs.json, trials.csv, recipe.json
outputs/final/<model>/<run>/seed-<s>/         results.json, history.csv, curves.png, predictions.csv, final.pt, config.yaml
                                              (LightGBM: best.txt, MLP: best.pt instead of final.pt)
outputs/final/<model>/<run>/                  batch_summary_final.csv, seed_summary_final.csv
outputs/logs/<timestamp>/                     overrides of one launch; on Cineca the job logs
```

`<run>` = `<model>_mp-<mp>_readout-<readout>_dir-in[_enc-rwpe16][_gfp-tuned]`; tabular models:
`lightgbm_readout-full`, `mlp_readout-full`. `results.json` holds the seven metrics (F1, precision, recall, PR-AUC,
ROC-AUC, precision and recall in the top 5 %) on train, val and test at the validation-chosen threshold, plus
`threshold`, `best_epoch`, `params`, `invariant_params` and the hyperparameters.

## Protocol

| Fixed architecture (`hyperparameters/gnn/fixed.yaml`) | |
|---|---|
| Layers, hidden, dropout | 2, 128, 0.3 |
| Neighbours, batch | [100, 100], 8,192 edges |
| Optimiser, epochs | Adam 1e-3 cosine, 20 epochs, pos_weight 8 |
| `invariant_params` (identical within a family) | GIN 67,587 · GATv2 67,585 · PNA 427,265 · Transformer 133,121 |

- Threshold swept on validation every epoch (best F1); best epoch = best validation PR-AUC; test scored exactly
  once with that checkpoint and threshold; every metric on all three splits.
- GNN search (`hyperparameters/gnn/space.yaml`, protocol of Luo et al. 2024): 100 trials × 2 seeds per operator
  and feature configuration (c2, c4), 100 epochs at a constant learning rate, no early stopping; best epoch = argmax
  of the seed-mean validation PR-AUC, winner = highest mean validation PR-AUC. Final retrain on train + val at the
  frozen epoch, 5 seeds, threshold = mean of the two tuning thresholds. Trials are drawn deterministically from the
  yaml space; a trial that does not fit in GPU memory is recorded `oom` and cannot win.
- Cross-checks (`hyperparameters/lightgbm/`, `hyperparameters/mlp/`): the 93 inputs the GNN readout sees (81 edge
  columns + 6 + 6 entity one-hots), train rows only, same threshold rule. LightGBM: successive halving (100
  configurations, eta 2), every fit early-stopped on validation PR-AUC. MLP: 100 random trials, best epoch by
  validation PR-AUC, patience 10.

## Documents

- `EXPERIMENTS.md` — status and pointers of every experiment run from this repository.
- `CLAUDE.md` — the rules of this repository (naming, configuration, protocol, verification).
- `../gnn_for_fraudulent_patterns` — the notebook-era thesis repository: finished Kaggle runs, `Progress_Report.md`.
- `LICENSE` — MIT.
