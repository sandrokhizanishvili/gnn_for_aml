# CLAUDE.md — gnn_for_aml

The thesis pipeline of `../gnn_for_fraudulent_patterns` rebuilt as one small, plain-Python repository (configuration
by Hydra), runnable on Cineca by the co-supervisor with `git pull` + one command per step, and on a laptop or Kaggle
for single runs. Anyone starting from the two raw IBM files must get the thesis numbers from this repo alone.
`README.md` is the user manual; `EXPERIMENTS.md` tracks what ran from here. The sister repository is **read-only
reference**: its `Data/` is the reference output of `prepare_data.py`, its `CLAUDE.md` holds the thesis rules this
repo inherits (fixed architecture, evaluation protocol, leakage rules, writing style).

## Rules

- **Simple and small.** Plain functions; three entry points (`prepare_data.py` once, `run.py` for every run,
  `aggregate.py` after a GNN search); one package `pipeline/` with one folder per model (`gnn`, `lightgbm`, `mlp`),
  each holding `model.py`, `train.py`, `tune.py`; YAML for every knob. No class hierarchies, no decorators beyond
  `@hydra.main`, no test-only flags, no code read from yaml at import time. Keep the Python near its current size
  (~1,700 lines); delete code rather than add options. Asserts only where a wrong value would silently corrupt a
  result. Docstrings of one line (two where the why needs it); comments only for a non-obvious why.
- **Names say what the thing holds**, in plain words: `prepare_data.py`, `feature_config/`, `machine/leonardo.yaml`,
  `tuning/gnn/search.yaml`, `node_encoding`, `gfp_sheet`. No tool names, no paper authors, no version tags, no
  abbreviations an outsider cannot expand. Thesis vocabulary readers will search for is kept: `recipe.json` (the
  frozen winner), `c1..c7`, the run name `<model>_mp-<mp>_readout-<ro>_dir-in[...]`, `fx_rates.csv`. Every yaml
  starts with one comment line saying what it is and which mode or EXPERIMENTS.md block uses it.
- **Configuration.** `hyperparameters/<model>/fixed.yaml` = the values of mode train; `search.yaml` = `fixed` + what
  the search changes + the `search:` block (values and the way they are drawn live in these files only, never in
  Python). A tabular model file selects its own hyperparameters file (`override /hyperparameters`), so
  `model=lightgbm,mlp` pairs each with its values. `tuning/<model>/{search,final}.yaml` are the saved commands.
- **`prepare_data.py` reproduces the thesis data files bit for bit** (full rebuild compared file by file on
  30 Oct 2026, `EXPERIMENTS.md` "Data preparation"). The exchange rates are frozen in `conf/fx_rates.csv` and must be
  read with `float_precision='round_trip'` (the default parser is one bit off, which changes the GFP amount
  statistics). `data/` holds the thesis files and is what every run uses. Do not change an operation in
  `prepare_data.py` without redoing that comparison.
- **Protocol is fixed** (README section 5). Threshold on validation, test scored once, every metric on three splits,
  `invariant_params` identical within a family. The restructure of 30 Oct 2026 was checked byte-identical against
  the previous `aml_hydra` code on a data slice (every mode, every model); the one deliberate change since is the
  best-epoch rule (validation PR-AUC for every model, `EXPERIMENTS.md` Log). Do not change `pipeline/gnn/model.py`,
  an epoch loop or a search without redoing a slice regression.
- **Every run is idempotent and resumable**: a done marker (`results.json` / `done.json` / `recipe.json`) returns at
  once; `resume.pt` after every epoch; `trials.csv` for the tabular searches. The cluster relies on it (requeue after
  the 24 h limit, re-running a multirun submits only what is missing).
- **Tuning never reads the test snapshot** (`splits=('train', 'val')`), and `done.json` / `recipe.json` record
  `test_split_loaded`.
- **Nothing on the cluster is edited by hand**: settings come from `conf/`, the account from `$ACCOUNT`.
- **Local check before any hand-off**: a cumulative data slice in the session scratchpad (the last 40k edges of each
  window, earlier windows as context), `paths.data=<slice> paths.outputs=<scratch>`, every mode, then a multirun
  (`-m`, in-process). Delete scratch outputs; never report their numbers. GNN training on the full data needs a GPU:
  Kaggle (README section 2) or Cineca.
- **Git**: never commit or push; propose a one-line commit message and wait. `data/`, `outputs/`, `*.pt`,
  `predictions.csv` stay out.
- Writing style of docs: plain English, short sentences, numbers in tables, no filler.
