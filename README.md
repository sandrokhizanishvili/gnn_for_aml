# GNNs for money-laundering detection

Master thesis (Sapienza): do engineered features help graph neural networks find money-laundering patterns in
transaction networks? Four operators (GIN, PNA, GATv2, graph Transformer) share one template; the knobs are which
edge features enter message passing and the readout (`feature_config`), a node encoding, the GFP sheet and the
hyperparameters. Two models without message passing (LightGBM, MLP) are the cross-check on the same inputs.
Dataset: IBM AML HI-Small, edge classification on cumulative temporal snapshots.

One config = one run: `python run.py` with overrides trains one model; `python run.py -m` runs every combination
of the listed values, one job each, in this process on a laptop or Kaggle, or one SLURM task each on Cineca.
The configuration tool is Hydra; everything else is plain Python.

```
prepare_data.py      data/raw (the two IBM files) -> data/ (graphs, RWPE encodings, the tuned GFP sheet)
run.py               one run: mode train | tune | final | check, any model
aggregate.py         after a GNN search: scores the trials, freezes each winner into recipe.json, summaries, checks
conf/
  config.yaml        one run; groups: hyperparameters, model, feature_config, machine, tuning
  prepare_data.yaml  data preparation knobs; fx_rates.csv = the USD rates of the thesis (frozen: Cineca is offline)
  model/             gin pna gat transformer lightgbm mlp
  hyperparameters/   <model>/fixed.yaml = the values of mode train; <model>/search.yaml = fixed + what the search changes
  feature_config/    c1 .. c7: which edge columns enter message passing (mp) and the readout
  machine/           local kaggle leonardo: data and output folders; leonardo adds the SLURM launcher
  tuning/            <model>/search.yaml and <model>/final.yaml: the saved search and final-retrain commands
pipeline/
  data.py            the snapshots, column sets, node encodings, the tuned GFP sheet, the train+val graph, the 93-column matrix
  gfp.py             GFP normalisation; the snapml stream used by prepare_data.py
  metrics.py         threshold sweep, the seven metrics, the per-epoch row, threshold sensitivity
  results.py         run names and folders, results.json, predictions.csv, curves.png, config.yaml, trial logs
  gnn/               model.py (the template; only the conv differs) train.py (loaders, epoch loop, resume) tune.py (the random search)
  lightgbm/          model.py train.py tune.py (successive halving)
  mlp/               model.py train.py tune.py (random search)
```

## 1. Setup

```bash
git clone <this repo> gnn_for_aml && cd gnn_for_aml
conda env create -p <env path> -f environment.yml        # torch 2.8 + CUDA 12.6, PyG 2.8.1, LightGBM, Hydra
```

**Data.** `data/` is not versioned. Either copy the prepared files (`train/val/test_graph.pt`, `feature_meta.json`,
`rwpe/`, `gfp_variants/`) into it, or build them from the raw files: put `HI-Small_Trans.csv` and
`HI-Small_accounts.csv` (Kaggle, "IBM Transactions for Anti Money Laundering") in `data/raw/` and run

```bash
python prepare_data.py                 # graphs -> rwpe -> gfp_tuned, hours on CPU; a finished step is skipped
python prepare_data.py step=rwpe       # one step
```

The GFP step needs IBM `snapml`, whose Windows build cannot create the Graph Feature Preprocessor: on Windows
`prepare_data.yaml` runs `pipeline/gfp.py` inside WSL (`python3 -m venv ~/gfp_env && ~/gfp_env/bin/pip install
'numpy<2' snapml` once); on Linux set `gfp.python=''` and `pip install snapml`. A rebuild reproduces the thesis
files bit for bit (`EXPERIMENTS.md`, "Data preparation").

## 2. Where to run

**Laptop.** `machine=local` is the default: `data/` and `outputs/` inside the repo, GPU if torch sees one, else CPU
(fine for LightGBM, the MLP and short checks; a GNN run needs a GPU). Multiruns (`-m`) run one job after another.

**Kaggle (GNN runs).** `notebooks/kaggle_fixed_architecture.ipynb` is the ready notebook (set `MODEL`, Run All). The prepared files are already Kaggle datasets: `sandrokhizanishvili/hi-small-gnn` (the
graphs and `feature_meta.json`), `hi-small-rwpe` and `hi-small-gfp-tuned`. Add them as inputs of a GPU notebook, then:

```
!pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse pyg_lib
!pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
!pip install torch_geometric lightgbm==4.6.0 hydra-core==1.3.2
!git clone <this repo> gnn_for_aml
%cd gnn_for_aml
!python run.py model=gin feature_config=c5 machine=kaggle
```

`machine=kaggle` reads the graphs from the `hi-small-gnn` dataset. The node-encoding and tuned-GFP studies also
need `rwpe/` and `gfp_variants/` next to the graphs; link the three datasets into one folder once per session:

```
!mkdir -p /kaggle/working/data && ln -sf /kaggle/input/datasets/sandrokhizanishvili/hi-small-gnn/* /kaggle/working/data/
!ln -sfn /kaggle/input/datasets/sandrokhizanishvili/hi-small-rwpe /kaggle/working/data/rwpe
!ln -sfn /kaggle/input/datasets/sandrokhizanishvili/hi-small-gfp-tuned /kaggle/working/data/gfp_variants
!python run.py -m model=gin,pna,gat,transformer feature_config=c4,c5 node_encoding=rwpe16 machine=kaggle paths.data=/kaggle/working/data
```

Any command of section 3 works the same way with `machine=kaggle` added. Results land in
`/kaggle/working/outputs/`; download that folder (or zip it without `*.pt` and `predictions.csv`) and unpack it into
`outputs/` locally. A finished run is skipped on a rerun, so a multirun that outlives the session continues where it
stopped. The searches (1,600 GNN jobs) are Cineca work.

**Cineca (Leonardo Booster).** Data in `$WORK/aml_thesis/data`, the environment in `$WORK/aml_thesis/env`
(`conda env create -p $WORK/aml_thesis/env -f environment.yml` on a login node), results in
`$WORK/aml_thesis/outputs`. Put your project account once in your shell: `echo 'export ACCOUNT=<account>' >> ~/.bashrc`
(`saldo -b` lists it). Every command is typed on a login node from the repo folder with `machine=leonardo`; the
jobs go to SLURM by themselves (partition `boost_usr_prod`, one GPU per task, 24 h, up to
`hydra.launcher.array_parallelism` tasks at once; lower it with `hydra.launcher.array_parallelism=8` if your
per-user GPU limit is smaller). Job logs: `outputs/logs/<timestamp>/jobs/`.

## 3. Commands

Every block of `EXPERIMENTS.md` is one line. A single run is the same line without the lists.

| Command | What |
|---|---|
| `python run.py mode=check` | environment test: versions, the device, one forward / backward pass per operator |
| `python run.py model=gin feature_config=c5` | one run at the fixed hyperparameters (`hyperparameters/gnn/fixed.yaml`) |
| `python run.py -m model=gin,pna,gat,transformer feature_config=c1,c2,c3,c4,c5,c6,c7` | the fixed-architecture grid (28 runs; add `seed=42,43,44,45,46` for a seed sweep) |
| `python run.py -m model=gin,pna,gat,transformer feature_config=c4,c5 node_encoding=rwpe16` | the node-encoding study (8 runs) |
| `python run.py -m model=gin,pna,gat,transformer feature_config=c4,c5 gfp_sheet=tuned` | the tuned-GFP study (8 runs; needs `gfp_variants/`) |
| `python run.py -m model=lightgbm,mlp seed=42,43,44,45,46` | the cross-check without message passing at the frozen thesis winners (10 runs) |
| `python run.py -m tuning=gnn/search machine=leonardo` | the GNN random search: 100 trials x 2 seeds x (4 operators x configs 2, 4) = 1,600 jobs |
| `python aggregate.py machine=leonardo` | scores the finished trials, checks them, freezes each pair's winner into `recipe.json` |
| `python run.py -m tuning=gnn/final machine=leonardo` | the 5-seed retrain of every frozen winner (40 jobs) |
| `python run.py -m tuning=lightgbm/search` then `tuning=lightgbm/final` | the LightGBM halving search (one job), then its winner x 5 seeds; same with `mlp` |

Runs are idempotent: a finished run returns at once, an interrupted one continues from `resume.pt`, a job cut by the
24 h limit is requeued by SLURM and resumes. So **run the same command again** until everything is done
(`squeue -u $USER` shows what is queued or running).

`fixed` vs `search` (`conf/hyperparameters/<model>/`):

| File | Holds | Used by |
|---|---|---|
| `fixed.yaml` | GNNs: the a-priori fixed architecture of the thesis. LightGBM, MLP: the frozen winner of the thesis search | mode train |
| `search.yaml` | `fixed` + what the search changes + `search:` (space, trials, seeds) | modes tune and final, `aggregate.py` |

## 4. Outputs

```
outputs/train/<model>/<run>/seed-<s>/        results.json, history.csv, curves.png, predictions.csv, best.pt (LightGBM: best.txt), config.yaml
outputs/tuning/<model>/<run>/t<id>/seed-<s>/ done.json, history.csv                 (GNN trials: no test loader, no checkpoints)
outputs/tuning/<model>/<run>/                t<id>/trial.json, recipe.json, trials_summary.csv, tuning_curves.png
                                             LightGBM, MLP: configs.json, trials.csv, recipe.json
outputs/final/<model>/<run>/seed-<s>/        results.json, history.csv, curves.png, predictions.csv, final.pt, config.yaml
outputs/final/<model>/<run>/                 batch_summary_final.csv, seed_summary_final.csv
outputs/logs/<timestamp>/                    the overrides of one launch; on Cineca the job logs
```

`<run>` = `<model>_mp-<mp>_readout-<readout>_dir-in[_enc-rwpe16][_gfp-tuned]` (`lightgbm_readout-full`,
`mlp_readout-full`). `results.json` holds the seven metrics (F1, precision, recall, PR-AUC, ROC-AUC, precision and
recall in the top 5 %) on train, val and test at the validation-chosen threshold, plus `threshold`, `best_epoch`,
`params`, `invariant_params` and the hyperparameters.

To send results back from Cineca (from `$WORK/aml_thesis`):
```bash
tar -czf results_small.tar.gz --exclude='*.pt' --exclude='predictions.csv' --exclude='logs' outputs
tar -czf results_predictions.tar.gz $(find outputs/final -name predictions.csv)
```

## 5. Protocol in short

- Fixed architecture (`hyperparameters/gnn/fixed.yaml`): 2 layers, hidden 128, dropout 0.3, [100, 100] neighbours,
  batch 8,192, Adam 1e-3 cosine, 20 epochs, pos_weight 8. Only the edge-projection and readout widths change between
  feature configurations; `invariant_params` must be identical within a family (GIN 67,587, GATv2 67,585,
  PNA 427,265, Transformer 133,121).
- Threshold swept on validation every epoch (best F1); best epoch = the best validation PR-AUC (`select_metric`;
  the Kaggle runs of the sister repository used F1); test scored exactly once with that checkpoint and threshold.
  Every metric on all three splits.
- GNN search (`hyperparameters/gnn/search.yaml`, protocol of Luo et al. 2024): 100 epochs at a constant LR, no early
  stopping, best epoch = argmax of the seed-mean validation PR-AUC, winner = highest mean validation PR-AUC; the final
  retrain uses train + val, the frozen epoch and the mean of the two tuning thresholds. Trials are drawn
  deterministically from the yaml space (seed 20261022); a trial that does not fit in GPU memory is recorded `oom`
  and cannot win.
- Cross-checks without message passing (`hyperparameters/lightgbm/`, `hyperparameters/mlp/`): the 93 inputs the GNN
  readout sees (81 edge columns + the 6 + 6 entity one-hots), train rows only, threshold swept on validation, test
  once. LightGBM: successive halving (100 configurations, eta 2, first round on the most recent 10 % of the train
  rows), every fit early-stopped on validation PR-AUC, ranked by validation PR-AUC; MLP: 100 unique uniform draws,
  Adam at a constant LR, best epoch by validation PR-AUC, patience 10. `invariant_params` is 0 for both.
