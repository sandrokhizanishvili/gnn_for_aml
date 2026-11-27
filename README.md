# Graph neural networks for anti-money-laundering detection

Master thesis, Sapienza University of Rome.

**Question.** Do engineered graph features help graph neural networks (GNNs) find money laundering in a network of
bank transactions?

**Task.** Each transaction (an edge between two accounts) is labelled laundering or not. Data: IBM AML HI-Small
(Altman et al., NeurIPS 2023), split in time 60 / 20 / 20 into train, validation and test.

**What is compared.**

| Variable | Values |
|---|---|
| GNN operator (`model`) | `gin`, `pna`, `gat` (GATv2), `transformer` (graph Transformer); all share one model template |
| Feature configuration (`feature_config`) | `c1`..`c7`: which edge columns enter message passing and the final classifier (table below) |
| Node encoding (`node_encoding`) | `none` or `rwpe16` (random-walk positional encoding) |
| GFP sheet (`gfp_sheet`) | `default` or `tuned` settings of IBM's Graph Feature Preprocessor (GFP) |
| Hyperparameters | one fixed architecture, then a search with one strategy for every model |
| Baselines without a graph | LightGBM and an MLP on the same 93 inputs |

| Config | Message passing | Final classifier (readout) |
|---|---|---|
| c1 | none | baseline |
| c2 | baseline | baseline |
| c3 | none | baseline + GFP |
| c4 | baseline + GFP | baseline + GFP |
| c5 | baseline | baseline + GFP |
| c6 | baseline | GFP only |
| c7 | none | GFP only |

"Baseline" = the 20 original transaction columns; "GFP" = the engineered graph features.

## Repository structure

```
prepare_data.py                   raw IBM files -> data/
run.py                            one run (mode train | tune | final | check)
aggregate.py                      during / after a GNN search: progress, checks, summaries
conf/
  config.yaml                     defaults of one run
  prepare_data.yaml               data preparation settings
  fx_rates.csv                    frozen USD exchange rates
  model/                          a GNN file also holds the operator's own search parameters
    gin.yaml
    pna.yaml
    gat.yaml
    transformer.yaml
    gps.yaml                      GraphGPS side test (not in the grid or the search)
    lightgbm.yaml
    mlp.yaml
  feature_config/                 which edge columns enter message passing and the readout
    c1.yaml
    c2.yaml
    c3.yaml
    c4.yaml
    c5.yaml
    c6.yaml
    c7.yaml
  hyperparameters/
    search.yaml                   the search strategy every model shares
    gnn/
      fixed.yaml                  the fixed architecture (mode train)
      space.yaml                  the GNN search space shared by every operator
    lightgbm/
      space.yaml                  LightGBM settings + search space
    mlp/
      space.yaml                  MLP settings + search space
  machine/
    local.yaml                    this computer
    kaggle.yaml                   Kaggle paths
    leonardo.yaml                 Cineca paths + SLURM launcher
  tuning/                         saved commands: the model, its hyperparameters file, the mode, the sweep
    gnn/
      search.yaml                 the GNN search
      final.yaml                  the final retrain of its winners
    lightgbm/
      search.yaml
      final.yaml
    mlp/
      search.yaml
      final.yaml
pipeline/
  data.py                         load the snapshots, encodings, the 93-column matrix
  gfp.py                          GFP features (snapml stream) and their normalisation
  metrics.py                      threshold sweep, the seven metrics
  results.py                      output files: results.json, predictions.csv, curves.png
  tune.py                         the search every model shares (Optuna study, trial loop)
  gnn/
    model.py                      the GNN template; only the convolution differs
    train.py                      loaders, epoch loop, resume
    tune.py                       a trial's recipe, the checks of aggregate.py
  lightgbm/
    model.py
    train.py                      the final refit
    tune.py                       one trial
  mlp/
    model.py
    train.py
    tune.py                       one trial
notebooks/
  gnn_fixed_architecture_runs.ipynb      Kaggle GPU runs: one operator x c1..c7
  gnn_fixed_architecture_results.ipynb   tables and plots of the downloaded runs (outputs/train/)
  gnn_parameter_matched_gin_runs.ipynb          Kaggle GPU runs: GIN x c1..c7 at hidden 304 (PNA's parameter count)
  gnn_parameter_matched_gat_runs.ipynb          the same for GATv2, hidden 304
  gnn_parameter_matched_transformer_runs.ipynb  the same for the Transformer, hidden 236
  gnn_node_encoding_runs.ipynb           Kaggle GPU runs: the node-encoding study (rwpe16)
  gnn_tuned_gfp_runs.ipynb               Kaggle GPU runs: the tuned-GFP study
  mlp_runs.ipynb                         Kaggle GPU: the MLP search, then its final
  EDA.ipynb                              the transaction graph: size, self-loops, degrees (runs on a laptop)
  thesis_figures.mplstyle                the matplotlib style of every thesis figure
environment.yml                   conda environment
data/                             not versioned: raw/, the graphs, rwpe/, gfp_variants/
outputs/                          not versioned: train/, tuning/, final/, logs/
```

## Quick start

```bash
git clone https://github.com/sandrokhizanishvili/gnn_for_aml.git
cd gnn_for_aml
conda env create -f environment.yml
conda activate aml_thesis

python prepare_data.py                      # needs the two raw files in data/raw/ (below)
python run.py mode=check                    # setup test
python run.py model=gin feature_config=c5   # one run
```

The result is in `outputs/train/gin/gin_mp-base_readout-full_dir-in/seed-42/results.json`.

Data:
- Raw files: `HI-Small_Trans.csv` and `HI-Small_accounts.csv` from the Kaggle dataset "IBM Transactions for Anti
  Money Laundering". Ready-made data: Kaggle `sandrokhizanishvili/hi-small-gnn`, `hi-small-rwpe`, `hi-small-gfp-tuned`.
- The GFP step needs IBM `snapml`. Linux: `pip install snapml`, then `python prepare_data.py gfp.python=''`.
  Windows: it runs in WSL; set it up once with `python3 -m venv ~/gfp_env && ~/gfp_env/bin/pip install 'numpy<2' snapml`.
- Finished steps are skipped. The rebuilt files match the thesis data bit for bit.

## How runs work

- One command = one run: `python run.py` plus settings, e.g. `model=pna seed=43`.
- `-m` runs every combination: `-m model=gin,pna feature_config=c1,c2` = 4 runs.
- `mode`: `train` (fixed architecture), `tune` (search), `final` (retrain the search winner), `check` (setup test).
- If a run stops, run the same command again: finished runs are skipped, stopped ones continue.
- A search is one Optuna study per model (GNN: per model and config). Jobs can share a study with `worker=<n>`;
  never start the same worker number twice.
- To extend a finished study (e.g. a 10-trial pilot to 100): delete its `recipe.json` and run the search again.

## Experiments

| Experiment | Command | Output |
|---|---|---|
| Fixed-architecture grid (28 runs) | `python run.py -m model=gin,pna,gat,transformer feature_config=c1,c2,c3,c4,c5,c6,c7` | `outputs/train/<model>/` |
| Same grid, five seeds | add `seed=42,43,44,45,46` | `outputs/train/<model>/<run>/seed-<s>/` |
| Node-encoding study (8 runs) | `python run.py -m model=gin,pna,gat,transformer feature_config=c4,c5 node_encoding=rwpe16` | `outputs/train/<model>/*_enc-rwpe16/` |
| Tuned-GFP study (8 runs) | `python run.py -m model=gin,pna,gat,transformer feature_config=c4,c5 gfp_sheet=tuned` | `outputs/train/<model>/*_gfp-tuned/` |
| GNN search (8 studies, 4 workers each = 32 jobs) | `python run.py -m tuning=gnn/search machine=leonardo` | `outputs/tuning/<model>/<run>/` |
| One GNN study, one worker | `python run.py tuning=gnn/search model=gin feature_config=c2` | same |
| Progress, checks and summaries of the GNN search | `python aggregate.py machine=leonardo` | `trials.csv`, `tuning_curves.png` |
| Retrain the winners (40 jobs) | `python run.py -m tuning=gnn/final machine=leonardo` | `outputs/final/<model>/` |
| LightGBM: search, then the winner × 5 seeds | `python run.py -m tuning=lightgbm/search`, then `python run.py -m tuning=lightgbm/final` | `outputs/tuning/lightgbm/`, `outputs/final/lightgbm/` |
| MLP: the same with `mlp` (on Kaggle: `notebooks/mlp_runs.ipynb`) | `python run.py -m tuning=mlp/search`, then `python run.py -m tuning=mlp/final` | `outputs/tuning/mlp/`, `outputs/final/mlp/` |

## Where to run

| Machine | Use for | How |
|---|---|---|
| Laptop (default) | data preparation, LightGBM, MLP, `mode=check` | GPU if there is one, else CPU. GNN training needs a GPU. |
| Kaggle (`machine=kaggle`) | GNN runs, the MLP search | the notebooks below |
| Cineca Leonardo (`machine=leonardo`) | GNN search and final retrain | run the command on a login node; it submits the jobs |

**Kaggle.** Open a notebook in a GPU session, add the dataset `sandrokhizanishvili/hi-small-gnn` as input, Run All:
`gnn_fixed_architecture_runs.ipynb` (set `MODEL`), `gnn_node_encoding_runs.ipynb`, `gnn_tuned_gfp_runs.ipynb` (these
two also need `hi-small-rwpe` and `hi-small-gfp-tuned`), `mlp_runs.ipynb`. Download `/kaggle/working/outputs/` into
`outputs/`. Kaggle loses its files between sessions: to continue a search, put its `outputs/tuning/` back first.
The three `gnn_parameter_matched_<operator>_runs.ipynb` repeat the fixed-architecture runs with a larger `hidden`
(`hyperparameters.hidden=304` for GIN and GATv2, `236` for the Transformer), so that each operator has PNA's parameter
count within 4.1%. Their runs keep the fixed-architecture run names: download their `results_small.zip`, whose run
folders end in `_h-<hidden>`.

**Cineca (Leonardo Booster).** Activate the environment, then run the commands of "Experiments" with
`machine=leonardo`. Each job gets one GPU for up to 24 h; a job cut by the limit is requeued and continues.
Test the setup first with `python run.py -m mode=check machine=leonardo`: `-m` submits it as a GPU job (the login
nodes have no GPU), and what it printed must end with `OK`.

| Setting | Value |
|---|---|
| Data and outputs | `data/` and `outputs/` in the repository folder, as on a laptop: type the commands there |
| Data kept elsewhere | add `paths.data=<folder>` to the commands |
| Project account | your default; for another, `export ACCOUNT=<account>` in `~/.bashrc` |
| Fewer jobs at once | `hydra.launcher.array_parallelism=<n>` |
| Fewer workers per study | `worker=0` or `worker=0,1` |
| What a job printed | `outputs/logs/<timestamp>/jobs/<job>/<job>_0_log.out` (errors: `_log.err`) |

Bring results back, from the repository folder:
```bash
tar -czf results_small.tar.gz --exclude='*.pt' --exclude='predictions.csv' --exclude='logs' outputs
tar -czf results_predictions.tar.gz $(find outputs/final -name predictions.csv)
```

## Outputs

| Folder | Contents |
|---|---|
| `outputs/train/<model>/<run>/seed-<s>/` | `results.json`, `history.csv`, `curves.png`, `predictions.csv`, `best.pt`, `config.yaml` |
| `outputs/tuning/<model>/<run>/` | `study.log` (the study), `trials.csv` (one row per trial), `tuning_curves.png`, `recipe.json`; GNN and MLP: `t<number>/history.csv` per trial |
| `outputs/final/<model>/<run>/seed-<s>/` | as in `train/`, with `final.pt` (LightGBM: `final.txt`) |
| `outputs/final/<model>/<run>/` | `batch_summary_final.csv`, `seed_summary_final.csv` |
| `outputs/logs/<timestamp>/` | overrides of one launch; on Cineca also the job logs |

- `<run>` = `<model>_mp-<mp>_readout-<readout>_dir-in`, plus `_enc-rwpe16` or `_gfp-tuned` when used.
  LightGBM and MLP: `lightgbm_readout-full`, `mlp_readout-full`.
- `results.json` holds seven metrics on train, validation and test: F1, precision, recall, PR-AUC, ROC-AUC, and
  precision and recall in the top 5 %. It also stores the threshold, the best epoch and the hyperparameters.
- `recipe.json` is the frozen winner of a search: its values, its best epoch (LightGBM: round) and its validation
  threshold. It is written when the last trial of the study ends.
- `trials.csv` has one row per trial, pruned trials included: its values, state, score, best epoch (LightGBM: round),
  threshold, and the seven metrics on train and validation at that epoch.
- `t<number>/history.csv` holds the same metrics for every epoch of one GNN or MLP trial.
- A GNN trial scores validation only, so its rows have no train metrics: scoring the train rows takes about a third
  of an epoch and the search does not use it. `hyperparameters.search.score_train=true` turns it on.

## Evaluation protocol

**All runs**
- The decision threshold is the one with the best F1 on validation.
- The best epoch is the one with the best validation PR-AUC.
- The test set is scored once, with that epoch and threshold. Searches never load the test set.

**Fixed architecture** (`conf/hyperparameters/gnn/fixed.yaml`)

| Setting | Value |
|---|---|
| Layers, hidden size, dropout | 2, 128, 0.3 |
| Neighbours sampled, batch size | [100, 100], 8,192 edges |
| Training | Adam, learning rate 1e-3 with cosine decay, 20 epochs, positive-class weight 8 |
| Parameters shared by all configs of one operator | GIN 67,587 · GATv2 67,585 · PNA 427,265 · Transformer 133,121 |

**Search, the same for every model** (`conf/hyperparameters/search.yaml`, `pipeline/tune.py`)

| Setting | Value |
|---|---|
| Sampler | Optuna TPE (seed 42, multivariate), 5 random start-up trials per bracket |
| Pruner | Hyperband, reduction factor 3, 3 brackets; nothing else stops a trial (no early stopping) |
| Budget | 100 trials per study, the start trial and the start-up trials included |
| A full trial | GNN and MLP: 100 epochs at a constant learning rate; LightGBM: 1,000 rounds. Pruner checks at 4, 12, 36 epochs / 40, 120, 360 rounds |
| Score of a trial | its best validation PR-AUC; that epoch / round, its best-F1 threshold and its metrics are kept (GNN: validation only) |
| Start trial | GNN: the fixed architecture; MLP: a hand-picked point. Never pruned |
| Seeds | one training seed (42) per trial; the winner is retrained with seeds 42-46 |
| Final | the winner is retrained on train + validation for its best epoch / round; the threshold stays the one chosen on validation in the search; test scored once per seed |

- Studies: LightGBM, MLP, and each GNN operator on c2 and on c4 (8 studies).
- A trial that runs out of GPU memory or diverges is recorded without a score and cannot win.
- `hyperparameters.mixed_precision=true` runs the model's forward pass in 16-bit numbers (weights and loss stay
  32-bit): less GPU memory and less time per epoch. Off by default; a winner is retrained in the precision of its search.

**GNN search space** (`conf/hyperparameters/gnn/space.yaml` + `space:` in `conf/model/<operator>.yaml`)

| Shared by every operator | Values |
|---|---|
| Learning rate | 0.0005, 0.001, 0.003, 0.005, 0.01 |
| Hidden size | 64, 128, 256, 512 |
| Message-passing layers | 1 to 10 (neighbours per hop: 100, 100, 50, 50, then 25) |
| Normalisation | none, batch, layer |
| Residual | identity, linear, none |
| Dropout | 0.2, 0.3, 0.5, 0.7 |
| Weight decay | 0, 1e-5, 5e-5, 5e-4, 1e-3, 1e-2 |
| Positive-class weight | 1 to 10 |
| Readout layers, readout width | 1, 2, 3; 0.5, 1, 2 x hidden |
| Activation | ReLU, PReLU, Swish |

| Operator | Its own parameters |
|---|---|
| GIN | `train_eps` true / false; node MLP layers 1 / 2; node MLP width 1 / 2 x hidden |
| GATv2 | heads 1 / 2 / 4 / 8; `share_weights` false / true; attention dropout 0 to 0.5. Self-loops off in the search |
| PNA | degree scalers on / off; towers 1 / 2 / 4; `divide_input` false / true |
| Transformer | heads 1 / 2 / 4 / 8; `beta` (gated skip) false / true; attention dropout 0 to 0.5 |

**LightGBM and MLP baselines** (`conf/hyperparameters/lightgbm/`, `conf/hyperparameters/mlp/`)
- Inputs: the 93 columns the GNN classifier sees (81 edge columns + 6 + 6 account-type one-hots).
- Search spaces: the two `space.yaml` files. The MLP uses the GNN values wherever a setting means the same.

## Documents

- `CLAUDE.md` — the rules of this repository (naming, configuration, protocol, verification).
- `LICENSE` — MIT.
