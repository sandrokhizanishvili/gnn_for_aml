'''One run. One config = one run; a multirun (-m) = one job per combination, in this process or one SLURM task each.

    python run.py model=gin feature_config=c5                       one fixed-architecture run (mode train)
    python run.py -m model=gin,pna feature_config=c1,c2,c3          a multirun
    python run.py -m tuning=gnn/search machine=leonardo             the GNN search: 4 workers per (model, feature_config)
    python run.py -m tuning=gnn/final machine=leonardo              the 5-seed retrain of every frozen winner
    python run.py -m tuning=lightgbm/search                         the LightGBM search (one job; same with mlp)
    python run.py -m tuning=lightgbm/final                          its frozen winner (recipe.json) x 5 seeds
    python run.py mode=check machine=leonardo                       environment test on a GPU node

A finished run (results.json / recipe.json) returns at once; an interrupted run continues from resume.pt, an
interrupted search from its study.log.
'''
import gc
import importlib.metadata as metadata
import json
import os
import platform

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf

from pipeline import data, metrics, results, tune
from pipeline.gnn import model as gnn_model, train as gnn_train, tune as gnn_tune
from pipeline.lightgbm import model as lgb_model, train as lgb_train, tune as lgb_tune
from pipeline.mlp import model as mlp_model, train as mlp_train, tune as mlp_tune


def recipe_of(hp):
    '''The GNN knobs of a hyperparameters file (without `for` and `search`).'''
    return gnn_model.check_recipe({k: hp[k] for k in gnn_model.RECIPE_KEYS})


# ---------------------------------------------------------------- GNN modes

def build(cfg, fit_graph, node_dim, mp_idx, readout_idx, r):
    '''Seeded model build on the graph trained on; prints the parameter fingerprint.'''
    data.set_seed(cfg['seed'])
    operator, local = cfg['model']['name'], cfg['model'].get('local', 'gin')
    deg_hist = data.in_degree_histogram(fit_graph) if 'pna' in (operator, local) else None
    model = gnn_model.build_model(operator, node_dim, len(mp_idx), len(readout_idx), r, deg_hist, local)
    n_params, n_invariant = gnn_model.count_params(model), gnn_model.invariant_params(model)
    print(f'MP edge features: {len(mp_idx)} | readout edge features: {len(readout_idx)} | node features: {node_dim}')
    print(f'total params: {n_params:,} | architecture-invariant: {n_invariant:,} | device {gnn_model.device}')
    return model, n_params, n_invariant


def train_gnn(cfg, name, out_dir, graphs, col_sets, node_dim, r):
    '''Train on the train graph, best validation epoch, test scored once at the validation-chosen threshold.'''
    mp_idx, readout_idx = col_sets[cfg['feature_config']['mp']], col_sets[cfg['feature_config']['readout']]
    model, n_params, n_invariant = build(cfg, graphs['train'], node_dim, mp_idx, readout_idx, r)
    loaders = gnn_train.make_loaders(graphs, mp_idx, readout_idx, r)
    history, best_epoch, threshold = gnn_train.train_one_run(model, loaders, out_dir, r)
    model.load_state_dict(torch.load(f'{out_dir}/best.pt', map_location=gnn_model.device))
    scores, preds = gnn_train.score_all_splits(model, loaders, graphs, threshold, cfg['seed'])
    results.save_predictions(out_dir, graphs, preds, threshold)
    results.plot_curves(history, f"{name} seed {cfg['seed']}", f'{out_dir}/curves.png', best_epoch,
                        test=(preds['test']['y'], preds['test']['prob'], scores['test']['pr_auc']))
    os.remove(f'{out_dir}/resume.pt')
    results.save_run(out_dir, results.result_row(name, cfg, best_epoch, threshold, n_params, n_invariant, scores), r)


def tune_gnn(cfg, graphs, col_sets, node_dim, r, trial_dir, report):
    '''One trial of the search: the recipe on the train graph, val scored every epoch (train too when
    search.score_train), no test loader, no checkpoints. Returns (best validation PR-AUC, its epoch, threshold and
    metrics); the score is None when the trial ran out of memory or diverged, so it cannot win.'''
    print(f'recipe {r}')
    os.makedirs(trial_dir, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()   # the memory of the previous trial
    mp_idx, readout_idx = col_sets[cfg['feature_config']['mp']], col_sets[cfg['feature_config']['readout']]
    model, n_params, n_invariant = build(cfg, graphs['train'], node_dim, mp_idx, readout_idx, r)
    try:
        loaders = gnn_train.make_loaders(graphs, mp_idx, readout_idx, r, splits=('train', 'val'))
        history, _, _ = gnn_train.train_one_run(model, loaders, trial_dir, r, save_best=False, report=report,
                                                score_train=cfg['hyperparameters']['search']['score_train'])
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        if not ('out of memory' in str(err).lower() or 'ALLOC_FAILED' in str(err)):
            raise
        print('OUT OF MEMORY: recorded, the trial cannot win')
        return None, {'error': 'out of memory'}
    if len(history) < r['epochs']:
        return None, {'error': f'diverged at epoch {len(history) + 1}'}
    best = metrics.best_epoch_scores(history)
    return best['val_pr_auc'], dict(best, params=n_params, invariant_params=n_invariant)


def final_gnn(cfg, name, out_dir, graphs, col_sets, node_dim, frozen):
    '''Retrain on train + val for the frozen epoch, score every split once at the frozen threshold; results.json last.'''
    r, stop_epoch, threshold = frozen['recipe'], frozen['best_epoch'], frozen['threshold']
    mp_idx, readout_idx = col_sets[cfg['feature_config']['mp']], col_sets[cfg['feature_config']['readout']]
    fit_graph = data.make_trainval_graph(graphs)
    model, n_params, n_invariant = build(cfg, fit_graph, node_dim, mp_idx, readout_idx, r)
    fit_loader = gnn_train.make_loader(fit_graph, True, mp_idx, readout_idx, r)
    eval_loaders = gnn_train.make_loaders(graphs, mp_idx, readout_idx, r, splits=('train', 'val'))
    history = gnn_train.train_fixed_epochs(model, fit_loader, out_dir, r, stop_epoch, eval_loaders)
    del fit_loader, fit_graph, eval_loaders
    gc.collect()
    loaders = gnn_train.make_loaders(graphs, mp_idx, readout_idx, r)
    scores, preds = gnn_train.score_all_splits(model, loaders, graphs, threshold, cfg['seed'])
    sensitivity = metrics.threshold_sensitivity(preds['test']['y'], preds['test']['prob'], threshold)
    seen_threshold, _ = metrics.best_f1_threshold(preds['val']['y'], preds['val']['prob'])   # val was trained on: secondary view
    seen = metrics.compute_metrics(preds['test']['y'], preds['test']['prob'], seen_threshold)
    print(f"test F1 {scores['test']['f1']:.4f} at the frozen threshold {threshold:.3f} | "
          f"{seen['f1']:.4f} at the seen-val sweep {seen_threshold:.3f}")
    results.save_predictions(out_dir, graphs, preds, threshold)
    results.plot_curves(history, f"{name} seed {cfg['seed']} (train+val retrain)", f'{out_dir}/curves.png',
                        test=(preds['test']['y'], preds['test']['prob'], scores['test']['pr_auc']))
    os.remove(f'{out_dir}/resume.pt')
    row = results.result_row(name, cfg, stop_epoch, threshold, n_params, n_invariant, scores)
    row.update(phase='retrain', trained_on='train+val', stop_epoch=stop_epoch, val_seen=True,
               val_seen_threshold=seen_threshold, **{f'seen_test_{m}': seen[m] for m in ('f1', 'precision', 'recall')})
    results.save_run(out_dir, row, r, extra={'threshold_sensitivity': sensitivity, 'trial_id': frozen['trial']})


# ---------------------------------------------------------------- tabular modes (lightgbm | mlp)

def tabular_inputs(cfg, graphs):
    '''The 93-column matrix X, the labels y and the row slices per split; for the MLP, X standardised and on the device.'''
    X, y, rows = data.load_matrix(graphs)
    if cfg['model']['name'] == 'mlp':
        X = torch.from_numpy(mlp_model.standardise(X, rows, cfg['hyperparameters']['std_floor'])).to(gnn_model.device)
    return X, y, rows


def final_tabular(cfg, name, out_dir, graphs, frozen):
    '''Refit the frozen winner on train + val for its frozen rounds / epochs, score every split once at the frozen
    threshold; results.json last.'''
    hp, op, seed = cfg['hyperparameters'], cfg['model']['name'], cfg['seed']
    params, stop, threshold = frozen['params'], frozen['best_epoch'], frozen['threshold']
    X, y, rows = tabular_inputs(cfg, graphs)
    if op == 'lightgbm':
        booster, history = lgb_train.fit_final_booster(params, hp, X, y, rows, stop, seed)
        prob, n_params = lgb_model.predict_booster(booster, X), stop
        booster.save_model(f'{out_dir}/final.txt')
        pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    else:
        fit_rows = slice(rows['train'].start, rows['val'].stop)
        model, history = mlp_train.fit_mlp(params, X, y, rows, fit_rows, stop, seed, out_dir)
        prob, n_params = mlp_model.predict_mlp(model, X, slice(0, len(y))), gnn_model.count_params(model)
        torch.save(model.state_dict(), f'{out_dir}/final.pt')
    scores, preds = metrics.score_splits(prob, y, rows, threshold)
    results.save_predictions(out_dir, graphs, preds, threshold)
    results.plot_curves(history, f'{name} seed {seed} (train+val refit)', f'{out_dir}/curves.png',
                        test=(preds['test']['y'], preds['test']['prob'], scores['test']['pr_auc']))
    row = results.result_row(name, dict(cfg, feature_config={'mp': 'none', 'readout': 'full'}), stop, threshold, n_params, 0, scores)
    row.update(mp_direction='none', phase='retrain', trained_on='train+val', stop_epoch=stop, val_seen=True)
    results.save_run(out_dir, row, dict(hp, point=params), {'trial_id': frozen['trial']})
    results.save_seed_summary(os.path.dirname(out_dir), results.finished_seeds(os.path.dirname(out_dir)))


# ---------------------------------------------------------------- the search (every model)

def search(cfg, out_dir, graphs, col_sets, node_dim):
    '''The search of one model (a GNN: on one feature configuration): trials until the study has n_trials, then
    recipe.json = the winner's values, its best epoch / round and its validation threshold.'''
    hp, model, seed = cfg['hyperparameters'], cfg['model'], cfg['seed']
    s = hp['search']
    if model['kind'] == 'tabular':
        X, y, rows = tabular_inputs(cfg, graphs)
        space, start, cap = s['space'], s.get('start'), hp['rounds' if model['name'] == 'lightgbm' else 'epochs']

        def run_trial(values, trial_dir, report):
            if model['name'] == 'lightgbm':
                return lgb_tune.run_trial(values, hp, X, y, rows, seed, report)
            return mlp_tune.run_trial(values, hp, X, y, rows, seed, report, trial_dir)
    else:
        base = recipe_of(hp)
        space, cap = dict(s['space'], **model['space']), base['epochs']
        start = {k: base[k] if k in base else gnn_model.CONV[model['name']][k] for k in space}   # the fixed architecture

        def run_trial(values, trial_dir, report):
            r = gnn_tune.trial_recipe(base, values, s['neighbours'])
            return tune_gnn(cfg, graphs, col_sets, node_dim, r, trial_dir, report)

    study = tune.open_study(out_dir, s, cap, cfg['worker'])
    tune.run_search(study, out_dir, space, start, s['n_trials'], cfg['worker'], run_trial)
    tune.save_summary(study, out_dir)
    if not all(t.state.is_finished() for t in study.trials):
        print('other workers still run trials of this study: the last one writes recipe.json')
        return
    best = study.best_trial
    frozen = dict(operator=model['name'], trial=best.number, params=best.params, val_pr_auc=best.value,
                  **{k: best.user_attrs[k] for k in ('best_epoch', 'threshold', 'val_f1')},
                  final_seeds=s['final_seeds'], test_split_loaded='test' in graphs)
    if model['kind'] == 'gnn':
        frozen.update(features=cfg['feature_config'], recipe=gnn_tune.trial_recipe(base, best.params, s['neighbours']))
    results.write_json_atomic(f'{out_dir}/recipe.json', frozen)
    print(f"winner: trial {best.number}, val PR-AUC {best.value:.4f} at epoch / round {frozen['best_epoch']} | {best.params}")


# ---------------------------------------------------------------- check and main

def check(cfg):
    '''Environment test: versions, the device, forward / backward passes of every operator on a synthetic graph.'''
    print(f'python {platform.python_version()} | torch {torch.__version__} | CUDA build {torch.version.cuda} | device {gnn_model.device}')
    for package in ('torch_geometric', 'torch_scatter', 'torch_sparse', 'lightgbm', 'optuna', 'hydra-core', 'hydra-submitit-launcher',
                    'numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib', 'psutil'):
        try:
            print(f'  {package} {metadata.version(package)}')
        except metadata.PackageNotFoundError:
            print(f'  {package} not installed' + (' (only needed on Cineca)' if 'submitit' in package else ''))
    assert cfg['machine'] == 'local' or torch.cuda.is_available(), 'no GPU visible to torch: check --gres=gpu:1 and the env'
    gen = torch.Generator().manual_seed(0)
    n_nodes, n_edges, cols = 300, 4000, list(range(81))
    graph = data.Data(x=torch.nn.functional.one_hot(torch.randint(0, 6, (n_nodes,), generator=gen), 6).float(),
                      edge_index=torch.randint(0, n_nodes, (2, n_edges), generator=gen),
                      edge_attr=torch.randn(n_edges, 81, generator=gen), edge_time=torch.arange(n_edges),
                      y=(torch.rand(n_edges, generator=gen) < 0.05).long(),
                      eval_mask=torch.ones(n_edges, dtype=torch.bool), num_nodes=n_nodes)
    r = dict(recipe_of(cfg['hyperparameters']), hidden=64, norm='batch', residual='linear', readout_layers=3, batch_size=512,
             num_neighbors=[10, 10])
    searched = {'gin': dict(train_eps=False, mlp_width=2), 'pna': dict(degree_scalers=False, towers=4, divide_input=True),
                'gat': dict(heads=8, share_weights=True, attention_dropout=0.5),
                'transformer': dict(heads=8, beta=True, attention_dropout=0.5), 'gps': {}}
    for op in gnn_model.OPERATORS:
        for conv, activation in (({}, 'relu'), (searched[op], 'prelu')):   # the fixed settings, then searched ones
            model = gnn_model.build_model(op, 6, 81, 81, dict(r, conv=conv, activation=activation),
                                          data.in_degree_histogram(graph) if op == 'pna' else None)
            loader, feats = gnn_train.make_loader(graph, True, cols, cols, r)
            logits, y = gnn_train.forward_batch(model, next(iter(loader)), feats)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            print(f'  {op:<11} {activation:<5} {conv or "fixed settings"} | params {gnn_model.count_params(model):>8,} | loss {loss.item():.4f}')
    print('OK')


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if cfg['mode'] == 'check':
        return check(cfg)
    hp, model, mode = cfg['hyperparameters'], cfg['model'], cfg['mode']
    assert hp['for'] == (model['name'] if model['kind'] == 'tabular' else 'gnn'), f"hyperparameters for {hp['for']}, model {model['name']}: lightgbm and mlp run as tuning=<model>/search and tuning=<model>/final"
    assert mode == 'train' or 'search' in hp, f"mode {mode} needs hyperparameters={hp['for']}/space"
    if model['kind'] == 'tabular':
        assert mode != 'train', f"{model['name']} has no fixed values: run tuning={model['name']}/search, then tuning={model['name']}/final"
        assert cfg['node_encoding'] == 'none', 'the tabular models use the 6 entity one-hots only'
        name = f"{model['name']}_readout-full"
    else:
        name = results.run_name(model['name'], cfg['feature_config'], cfg['node_encoding'], cfg['gfp_sheet'])
        if model.get('local', 'gin') != 'gin':
            name += f"_local-{model['local']}"
    out_dir = results.run_dir(cfg, name)
    if os.path.exists(f"{out_dir}/{'recipe.json' if mode == 'tune' else 'results.json'}"):
        print(f'{out_dir}: already finished, nothing to do')
        return
    print(f"=== {mode} {name} seed {cfg['seed']} -> {out_dir}")
    print(OmegaConf.to_yaml(cfg), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    results.save_config(cfg, out_dir)
    splits = ('train', 'val') if mode == 'tune' else data.SPLITS   # a search never reads the test snapshot
    graphs, col_sets, node_dim = data.load_graphs(cfg['paths']['data'], cfg['gfp_sheet'], splits)
    if model['kind'] == 'gnn':
        node_dim = data.set_node_encoding(graphs, cfg['node_encoding'], cfg['paths']['data'])

    if mode == 'train':
        return train_gnn(cfg, name, out_dir, graphs, col_sets, node_dim, recipe_of(hp))
    if mode == 'tune':
        return search(cfg, out_dir, graphs, col_sets, node_dim)
    recipe_path = f"{results.run_root(cfg['paths']['outputs'], 'tune', model['name'], name)}/recipe.json"
    assert os.path.exists(recipe_path), f'{recipe_path} not found: run the search first'
    frozen = json.load(open(recipe_path))
    if model['kind'] == 'tabular':
        final_tabular(cfg, name, out_dir, graphs, frozen)
    else:
        final_gnn(cfg, name, out_dir, graphs, col_sets, node_dim, frozen)


if __name__ == '__main__':
    main()
