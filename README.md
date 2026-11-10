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
| Hyperparameters | one fixed architecture, then a random search |
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
aggregate.py                      after a GNN search: scores the trials, freezes the winners
conf/
  config.yaml                     defaults of one run
  prepare_data.yaml               data preparation settings
  fx_rates.csv                    frozen USD exchange rates
  model/
    gin.yaml
    pna.yaml
    gat.yaml
    transformer.yaml
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
    gnn/
      fixed.yaml                  the fixed architecture (mode train)
      space.yaml                  the GNN search space
    lightgbm/
      space.yaml                  LightGBM settings + search space
    mlp/
      space.yaml                  MLP settings + search space
  machine/
    local.yaml                    this computer
    kaggle.yaml                   Kaggle paths
    leonardo.yaml                 Cineca paths + SLURM launcher
  tuning/                         saved commands
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
  gnn/
    model.py                      the GNN template; only the convolution differs
    train.py                      loaders, epoch loop, resume
    tune.py                       the random search
  lightgbm/
    model.py
    train.py
    tune.py                       successive halving
  mlp/
    model.py
    train.py
    tune.py                       the random search
notebooks/
  gnn_fixed_architecture_runs.ipynb      Kaggle GPU runs: one operator x c1..c7
  gnn_fixed_architecture_results.ipynb   tables and plots of the downloaded runs (outputs/train/)
environment.yml                   conda environment
EXPERIMENTS.md                    status of every experiment
data/                             not versioned: raw/, the graphs, rwpe/, gfp_variants/
outputs/                          not versioned: train/, tuning/, final/, logs/
```

## Quick start

```bash
# 1. Environment
git clone https://github.com/sandrokhizanishvili/gnn_for_aml.git && cd gnn_for_aml
conda env create -p <env path> -f environment.yml

# 2. Data: put HI-Small_Trans.csv and HI-Small_accounts.csv (Kaggle, "IBM Transactions for
#    Anti Money Laundering") in data/raw/, then
python prepare_data.py                  # builds graphs -> rwpe -> gfp_tuned; finished steps are skipped
python prepare_data.py step=rwpe        # or one step only

# 3. Check the setup, then train one model
python run.py mode=check                # versions, device, one pass per operator
python run.py model=gin feature_config=c5
```

The result is in `outputs/train/gin/gin_mp-base_readout-full_dir-in/seed-42/results.json`.

Notes on step 2:
- The GFP step needs IBM `snapml`. Linux: `pip install snapml` and set `gfp.python=''`. Windows: the step runs in
  WSL; set it up once with `python3 -m venv ~/gfp_env && ~/gfp_env/bin/pip install 'numpy<2' snapml`.
- The rebuilt files match the thesis data bit for bit (`EXPERIMENTS.md`, "Data preparation").
- Ready-made copies are on Kaggle: `sandrokhizanishvili/hi-small-gnn`, `hi-small-rwpe`, `hi-small-gfp-tuned`.

## How runs work

- One command = one run: `python run.py` plus overrides (`model=pna seed=43`).
- `-m` runs every combination of comma-separated values (`-m model=gin,pna feature_config=c1,c2` = 4 runs).
- Runs can be repeated safely: a finished run returns at once, an interrupted one resumes from `resume.pt`.
  If something stops, run the same command again.
- `mode` = `train` (fixed architecture), `tune` (search), `final` (retrain the search winner), `check` (setup test).

## Experiments

| Experiment | Command | Output |
|---|---|---|
| Fixed-architecture grid (28 runs) | `python run.py -m model=gin,pna,gat,transformer feature_config=c1,c2,c3,c4,c5,c6,c7` | `outputs/train/<model>/` |
| Same grid, five seeds | add `seed=42,43,44,45,46` | `outputs/train/<model>/<run>/seed-<s>/` |
| Node-encoding study (8 runs) | `python run.py -m model=gin,pna,gat,transformer feature_config=c4,c5 node_encoding=rwpe16` | `outputs/train/<model>/*_enc-rwpe16/` |
| Tuned-GFP study (8 runs) | `python run.py -m model=gin,pna,gat,transformer feature_config=c4,c5 gfp_sheet=tuned` | `outputs/train/<model>/*_gfp-tuned/` |
| GNN random search (800 jobs) | `python run.py -m tuning=gnn/search machine=leonardo` | `outputs/tuning/<model>/<run>/t<id>/` |
| Pick the search winners | `python aggregate.py machine=leonardo` | `outputs/tuning/<model>/<run>/recipe.json` |
| Retrain the winners (40 jobs) | `python run.py -m tuning=gnn/final machine=leonardo` | `outputs/final/<model>/` |
| LightGBM baseline: search, then winner × 5 seeds | `python run.py -m tuning=lightgbm/search`, then `python run.py -m tuning=lightgbm/final` | `outputs/tuning/lightgbm/`, `outputs/final/lightgbm/` |
| MLP baseline | same as LightGBM with `mlp` | `outputs/tuning/mlp/`, `outputs/final/mlp/` |

## Where to run

| Machine | Use for | How |
|---|---|---|
| Laptop (`machine=local`, default) | data preparation, LightGBM, MLP, `mode=check` | Uses the GPU if there is one, else CPU. GNN training needs a GPU. |
| Kaggle (`machine=kaggle`) | GNN runs | See below. |
| Cineca Leonardo (`machine=leonardo`) | GNN search and final retrain | Run the command on a login node; it submits SLURM jobs. See below. |

**Kaggle.** Open `notebooks/gnn_fixed_architecture_runs.ipynb` in a GPU notebook, add the dataset
`sandrokhizanishvili/hi-small-gnn` as input, set `MODEL`, Run All. For the node-encoding and tuned-GFP studies use
`notebooks/gnn_node_encoding_runs.ipynb` and `notebooks/gnn_tuned_gfp_runs.ipynb` (they also link `hi-small-rwpe`
and `hi-small-gfp-tuned`). Download `/kaggle/working/outputs/` into the local `outputs/`.

**Cineca (Leonardo Booster).** Jobs use partition `boost_usr_prod`, one GPU each, 24 h; a job that hits the limit
is requeued and resumes.

| Setting | Value |
|---|---|
| Folders | `$WORK/aml_thesis/data`, `$WORK/aml_thesis/env`, `$WORK/aml_thesis/outputs` |
| Environment | `conda env create -p $WORK/aml_thesis/env -f environment.yml` |
| Project account | your default project; to charge another, add `export ACCOUNT=<account>` to `~/.bashrc` (see `saldo -b`) |
| Fewer jobs at once | `hydra.launcher.array_parallelism=<n>` |
| Job logs | `outputs/logs/<timestamp>/jobs/` |

Bring results back (from `$WORK/aml_thesis`):
```bash
tar -czf results_small.tar.gz --exclude='*.pt' --exclude='predictions.csv' --exclude='logs' outputs
tar -czf results_predictions.tar.gz $(find outputs/final -name predictions.csv)
```

## Outputs

| Folder | Contents |
|---|---|
| `outputs/train/<model>/<run>/seed-<s>/` | `results.json`, `history.csv`, `curves.png`, `predictions.csv`, `best.pt`, `config.yaml` |
| `outputs/tuning/<model>/<run>/` | GNN: `t<id>/` per trial (`trial.json`, `seed-<s>/done.json`, `history.csv`), `recipe.json`, `trials_summary.csv`, `tuning_curves.png`. LightGBM, MLP: `configs.json`, `trials.csv`, `recipe.json` |
| `outputs/final/<model>/<run>/seed-<s>/` | as in `train/`, with `final.pt` (LightGBM: `best.txt`; MLP: `best.pt`) |
| `outputs/final/<model>/<run>/` | `batch_summary_final.csv`, `seed_summary_final.csv` |
| `outputs/logs/<timestamp>/` | overrides of one launch; on Cineca also the job logs |

- `<run>` = `<model>_mp-<mp>_readout-<readout>_dir-in`, plus `_enc-rwpe16` or `_gfp-tuned` when used.
  LightGBM and MLP: `lightgbm_readout-full`, `mlp_readout-full`.
- `results.json` holds seven metrics on train, validation and test: F1, precision, recall, PR-AUC, ROC-AUC, and
  precision and recall in the top 5 %. It also stores the threshold, the best epoch and the hyperparameters.
- `recipe.json` is the frozen winner of a search.

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

**GNN search** (`conf/hyperparameters/gnn/space.yaml`, protocol of Luo et al. 2024)
- 100 random trials per operator and feature configuration (c2, c4), one seed, 100 epochs, constant learning rate.
- Winner = highest validation PR-AUC. It is retrained on train + validation at its best epoch with 5 seeds.
- One seed per trial because the Cineca budget is 1,500 GPU hours; more seeds can be added in `tuning_seeds`.
- A trial that runs out of GPU memory is recorded as `oom` and cannot win.

**LightGBM and MLP baselines** (`conf/hyperparameters/lightgbm/`, `conf/hyperparameters/mlp/`)
- Inputs: the 93 columns the GNN classifier sees (81 edge columns + 6 + 6 account-type one-hots), train rows only.
- LightGBM: successive halving over 100 configurations (eta 2), each fit early-stopped on validation PR-AUC.
- MLP: 100 random trials, early stopping with patience 10.

## Documents

- `EXPERIMENTS.md` — status of every experiment run from this repository.
- `CLAUDE.md` — the rules of this repository (naming, configuration, protocol, verification).
- `../gnn_for_fraudulent_patterns` — the earlier, notebook-based thesis repository: finished Kaggle runs,
  `Progress_Report.md`.
- `LICENSE` — MIT.
