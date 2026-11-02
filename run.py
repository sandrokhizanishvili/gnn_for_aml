'''One run. One config = one run; a multirun (-m) = one job per combination, in this process or one SLURM task each.

    python run.py model=gin feature_config=c5                       one fixed-architecture run (mode train)
    python run.py -m model=gin,pna feature_config=c1,c2,c3          a multirun
    python run.py -m tuning=gnn/search machine=leonardo             every (model, feature_config, trial, seed) of the GNN search
    python run.py -m tuning=gnn/final machine=leonardo              the 5-seed retrain of every frozen winner
    python run.py -m tuning=lightgbm/search                         the LightGBM halving search (one job; same with mlp)
    python run.py -m tuning=lightgbm/final                          its frozen winner (recipe.json) x 5 seeds
    python run.py mode=check machine=leonardo                       environment test on a GPU node

A finished run (results.json / done.json / recipe.json) returns at once; an interrupted one continues from resume.pt.
'''
import gc
import importlib.metadata as metadata
import json
import os
import platform
import socket
import time

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf

from pipeline import data, metrics, results
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
    operator = cfg['model']['name']
    deg_hist = data.in_degree_histogram(fit_graph) if operator == 'pna' else None
    model = gnn_model.build_model(operator, node_dim, len(mp_idx), len(readout_idx), r, deg_hist)
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


def tune_gnn(cfg, name, out_dir, graphs, col_sets, node_dim, r, trial):
    '''One (trial, seed): every epoch of the trial's recipe on the train graph, train + val scored every epoch,
    no test loader, no checkpoints; done.json marks the seed finished (aggregate.py scores the trial).'''
    print(f"trial t{trial['id']:03d} {trial['tag']} | recipe {r}")
    mp_idx, readout_idx = col_sets[cfg['feature_config']['mp']], col_sets[cfg['feature_config']['readout']]
    model, n_params, n_invariant = build(cfg, graphs['train'], node_dim, mp_idx, readout_idx, r)
    t0 = time.time()
    try:
        loaders = gnn_train.make_loaders(graphs, mp_idx, readout_idx, r, splits=('train', 'val'))
        history, _, _ = gnn_train.train_one_run(model, loaders, out_dir, r, save_best=False)
        diverged_at = len(history) + 1 if len(history) < r['epochs'] else None   # no early stopping: fewer epochs = diverged
        info = dict(status='diverged' if diverged_at else 'ok', diverged_at=diverged_at, epochs_run=len(history))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        if not ('out of memory' in str(err).lower() or 'ALLOC_FAILED' in str(err)):
            raise
        print('OUT OF MEMORY: recorded, the trial cannot win')   # not retried: it would hit the same batch again
        info = dict(status='oom', diverged_at=None, epochs_run=0, error=str(err)[:200])
    if os.path.exists(f'{out_dir}/resume.pt'):
        os.remove(f'{out_dir}/resume.pt')
    results.write_json_atomic(f'{out_dir}/done.json', dict(
        info, trial_id=trial['id'], tag=trial['tag'], operator=cfg['model']['name'], features=cfg['feature_config'],
        seed=cfg['seed'], recipe=r, params=n_params, invariant_params=n_invariant, test_split_loaded='test' in graphs,
        minutes=(time.time() - t0) / 60, host=socket.gethostname(), job_id=os.environ.get('SLURM_JOB_ID')))
    print(f"seed {cfg['seed']}: {info['status']} -> {out_dir}/done.json")


def final_gnn(cfg, name, out_dir, graphs, col_sets, node_dim, frozen):
    '''Retrain on train + val for the frozen epoch, score every split once at the frozen threshold; results.json last.'''
    r, stop_epoch, threshold = frozen['recipe'], frozen['best_epoch'], frozen['thresholds'][str(cfg['seed'])]
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
    results.save_run(out_dir, row, r, extra={'threshold_sensitivity': sensitivity, 'trial_id': frozen['trial_id']})


# ---------------------------------------------------------------- tabular modes (lightgbm | mlp)

def tabular_inputs(cfg, graphs):
    '''The 93-column matrix X, the labels y and the row slices per split; for the MLP, X standardised and on the device.'''
    X, y, rows = data.load_matrix(graphs)
    if cfg['model']['name'] == 'mlp':
        X = torch.from_numpy(mlp_model.standardise(X, rows, cfg['hyperparameters']['std_floor'])).to(gnn_model.device)
    return X, y, rows


def tune_tabular(cfg, out_dir, graphs):
    '''The search of one tabular model -> recipe.json (the winner).'''
    hp, op = cfg['hyperparameters'], cfg['model']['name']
    X, y, rows = tabular_inputs(cfg, graphs)
    winner = lgb_tune.successive_halving(X, y, rows, hp, out_dir) if op == 'lightgbm' else mlp_tune.random_search(X, y, rows, hp, out_dir)
    print(f"winner: trial {winner['trial']} val PR-AUC {winner['val_pr_auc']:.4f} | {winner['params']}")
    results.write_json_atomic(f'{out_dir}/recipe.json', dict(winner, operator=op, search=hp['search'], test_split_loaded='test' in graphs))


def train_tabular(cfg, name, out_dir, graphs, point, extra):
    '''One fit at `point`, the frozen winner of recipe.json (mode final), scored on every split.'''
    hp, op, seed = cfg['hyperparameters'], cfg['model']['name'], cfg['seed']
    X, y, rows = tabular_inputs(cfg, graphs)
    if op == 'lightgbm':
        booster, history, best_epoch = lgb_train.fit_final_booster(point, hp, X, y, rows, seed)
        threshold = next(h['threshold'] for h in history if h['epoch'] == best_epoch)
        prob, n_params = lgb_model.predict_booster(booster, X, best_epoch), best_epoch
        booster.save_model(f'{out_dir}/best.txt', num_iteration=best_epoch)
    else:
        model, history, best_epoch = mlp_train.fit_mlp(point, hp, X, y, rows, seed)
        threshold = history[best_epoch - 1]['threshold']
        prob, n_params = mlp_model.predict_mlp(model, X, slice(0, len(y))), gnn_model.count_params(model)
        torch.save(model.state_dict(), f'{out_dir}/best.pt')
    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    scores, preds = metrics.score_splits(prob, y, rows, threshold)
    results.save_predictions(out_dir, graphs, preds, threshold)
    results.plot_curves(history, f'{name} seed {seed}', f'{out_dir}/curves.png', best_epoch,
                        test=(preds['test']['y'], preds['test']['prob'], scores['test']['pr_auc']))
    row = results.result_row(name, dict(cfg, feature_config={'mp': 'none', 'readout': 'full'}), best_epoch, threshold, n_params, 0, scores)
    row['mp_direction'] = 'none'
    results.save_run(out_dir, row, dict(hp, point=point), extra)
    if cfg['mode'] == 'final':
        results.save_seed_summary(os.path.dirname(out_dir), results.finished_seeds(os.path.dirname(out_dir)))


# ---------------------------------------------------------------- check and main

def check(cfg):
    '''Environment test: versions, the device, one forward / backward pass per operator on a synthetic graph.'''
    print(f'python {platform.python_version()} | torch {torch.__version__} | CUDA build {torch.version.cuda} | device {gnn_model.device}')
    for package in ('torch_geometric', 'torch_scatter', 'torch_sparse', 'lightgbm', 'hydra-core', 'hydra-submitit-launcher',
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
    for op in gnn_model.OPERATORS:
        model = gnn_model.build_model(op, 6, 81, 81, r, data.in_degree_histogram(graph) if op == 'pna' else None)
        loader, feats = gnn_train.make_loader(graph, True, cols, cols, r)
        logits, y = gnn_train.forward_batch(model, next(iter(loader)), feats)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        print(f'  {op:<11} params {gnn_model.count_params(model):>8,} | loss {loss.item():.4f}')
    print('OK')


@hydra.main(config_path='conf', config_name='config', version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if cfg['mode'] == 'check':
        return check(cfg)
    hp, model, mode = cfg['hyperparameters'], cfg['model'], cfg['mode']
    assert hp['for'] == (model['name'] if model['kind'] == 'tabular' else 'gnn'), f"hyperparameters for {hp['for']}, model {model['name']}"
    assert mode == 'train' or 'search' in hp, f"mode {mode} needs hyperparameters={hp['for']}/space"
    if model['kind'] == 'tabular':
        assert mode != 'train', f"{model['name']} has no fixed values: run tuning={model['name']}/search, then tuning={model['name']}/final"
        assert cfg['node_encoding'] == 'none', 'the tabular models use the 6 entity one-hots only'
        name = f"{model['name']}_readout-full"
    else:
        name = results.run_name(model['name'], cfg['feature_config'], cfg['node_encoding'], cfg['gfp_sheet'])
    out_dir = results.run_dir(cfg, name)
    marker = 'results.json' if mode != 'tune' else 'recipe.json' if model['kind'] == 'tabular' else 'done.json'
    if os.path.exists(f'{out_dir}/{marker}'):
        print(f'{out_dir}: already finished, nothing to do')
        return
    print(f"=== {mode} {name} seed {cfg['seed']} -> {out_dir}")
    print(OmegaConf.to_yaml(cfg), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    results.save_config(cfg, out_dir)
    splits = ('train', 'val') if mode == 'tune' else data.SPLITS   # tuning never reads the test snapshot
    graphs, col_sets, node_dim = data.load_graphs(cfg['paths']['data'], cfg['gfp_sheet'], splits)
    tuning_root = results.run_root(cfg['paths']['outputs'], 'tune', model['name'], name)

    if model['kind'] == 'tabular':
        if mode == 'tune':
            return tune_tabular(cfg, out_dir, graphs)
        frozen = json.load(open(f'{tuning_root}/recipe.json'))
        return train_tabular(cfg, name, out_dir, graphs, frozen['params'], {'trial_id': frozen['trial']})

    node_dim = data.set_node_encoding(graphs, cfg['node_encoding'], cfg['paths']['data'])
    r = recipe_of(hp)
    if mode == 'train':
        train_gnn(cfg, name, out_dir, graphs, col_sets, node_dim, r)
    elif mode == 'tune':
        assert cfg['trial'] is not None and 0 <= cfg['trial'] < hp['search']['n_trials'], 'tune needs trial=<0..n_trials-1>'
        trial = gnn_tune.draw_trials(hp['search'])[cfg['trial']]
        tune_gnn(cfg, name, out_dir, graphs, col_sets, node_dim, gnn_tune.trial_recipe(r, trial), trial)
    else:
        final_gnn(cfg, name, out_dir, graphs, col_sets, node_dim, gnn_tune.load_recipe(f'{tuning_root}/recipe.json'))


if __name__ == '__main__':
    main()
