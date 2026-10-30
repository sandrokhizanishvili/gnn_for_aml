'''The GNN random search (protocol of Luo et al. 2024): the deterministic trial draw, trial scoring from the seeds'
histories, the winner and recipe.json, the summary table and plot, the checks aggregate.py runs.'''
import glob
import itertools
import json
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.gnn.model import check_recipe
from pipeline.metrics import METRICS
from pipeline.results import write_json_atomic

matplotlib.use('Agg')
TRIAL_METRIC_KEYS = [f'{split}_{m}' for split in ('train', 'val') for m in METRICS + ('loss',)]
SHORT = {'lr': 'lr', 'hidden': 'h', 'dropout': 'd', 'weight_decay': 'wd', 'pos_weight': 'pw', 'readout_layers': 'ro',
         'readout_width': 'rw', 'num_layers': 'l', 'norm': '', 'residual': 'res'}


def draw_trials(search):
    '''The pre-registered draw: trial 0 = the fixed values (no knobs); trials 1..n-1 = cells of the space from a
    partial Fisher-Yates shuffle (random.Random(draw_seed)) plus a depth from a second stream (draw_seed + 1).
    Trial i never depends on n_trials. Returns [{'id', 'tag', 'knobs'}].'''
    space, depths, seed = search['space'], search['depths'], search['draw_seed']
    cells = list(itertools.product(*space.values()))
    order, rng = list(range(len(cells))), random.Random(seed)
    for i in range(search['n_trials'] - 1):
        j = i + int(rng.random() * (len(order) - i))
        order[i], order[j] = order[j], order[i]
    depth_rng = random.Random(seed + 1)
    trials = [{'id': 0, 'tag': 'base', 'knobs': {}}]
    for i in range(1, search['n_trials']):
        knobs = dict(zip(space, cells[order[i - 1]]))
        depth = depths[int(depth_rng.random() * len(depths))]
        knobs.update(num_layers=depth, num_neighbors=list(search['neighbours'][:depth]))
        tag = '_'.join(f'{SHORT[k]}{v:g}' if isinstance(v, (int, float)) else f'{SHORT[k]}{v}'
                       for k, v in knobs.items() if k != 'num_neighbors')
        trials.append({'id': i, 'tag': tag, 'knobs': knobs})
    return trials


def trial_recipe(base, trial):
    '''The trial's full recipe: the search file's values overlaid with the trial's knobs.'''
    return check_recipe(dict(base, **trial['knobs']))


def finalise_trial(trial_dir, seeds):
    '''trial.json of one trial once every seed has its done.json (None while a seed is missing): best epoch = argmax
    of the seed-mean val <select_metric>, per-seed thresholds and metrics at that epoch. oom / diverged before the
    first epoch leaves a trial without metrics (it cannot win).'''
    path = f'{trial_dir}/trial.json'
    if os.path.exists(path):
        return json.load(open(path))
    done_paths = [f'{trial_dir}/seed-{s}/done.json' for s in seeds]
    if not all(os.path.exists(p) for p in done_paths):
        return None
    done = {s: json.load(open(p)) for s, p in zip(seeds, done_paths)}
    first = done[seeds[0]]
    trial = {'trial_id': first['trial_id'], 'tag': first['tag'], 'operator': first['operator'], 'features': first['features'],
             'recipe': first['recipe'], 'seeds': list(seeds), 'params': first['params'],
             'invariant_params': first['invariant_params'], 'status': 'ok',
             'diverged_at': {str(s): done[s]['diverged_at'] for s in seeds if done[s]['diverged_at']} or None}
    if any(done[s]['status'] == 'oom' or not done[s]['epochs_run'] for s in seeds):
        trial['status'] = 'oom' if any(done[s]['status'] == 'oom' for s in seeds) else 'diverged'
    else:
        metric = first['recipe']['select_metric']
        histories = {s: pd.read_csv(f'{trial_dir}/seed-{s}/history.csv') for s in seeds}
        n_common = min(len(h) for h in histories.values())
        mean_curve = np.mean([h[f'val_{metric}'].to_numpy()[:n_common] for h in histories.values()], axis=0)
        best_epoch = int(np.argmax(mean_curve)) + 1
        per_seed = {}
        for s, h in histories.items():
            row = h.loc[h['epoch'] == best_epoch].iloc[0]
            per_seed[str(s)] = {k: float(row[k]) for k in TRIAL_METRIC_KEYS + ['threshold']}
            per_seed[str(s)]['epochs_run'] = int(len(h))
        trial.update(best_epoch=best_epoch, epochs_common=n_common, per_seed=per_seed,
                     thresholds={s: per_seed[s]['threshold'] for s in per_seed},
                     status='diverged' if trial['diverged_at'] else 'ok')
        trial[f'val_{metric}_mean_curve'] = mean_curve.tolist()
        for key in TRIAL_METRIC_KEYS:
            values = np.array([per_seed[str(s)][key] for s in seeds])
            trial[f'{key}_mean'], trial[f'{key}_std'] = float(values.mean()), float(values.std(ddof=1))
        trial['gap_pr_auc'] = trial['train_pr_auc_mean'] - trial['val_pr_auc_mean']
    write_json_atomic(path, trial)
    return trial


def choose_best_trial(trials, select_metric):
    '''Winner of one (operator, features): highest mean val <select_metric>, tie-break the other metric, then the id.'''
    other = 'f1' if select_metric == 'pr_auc' else 'pr_auc'
    scored = [t for t in trials if f'val_{select_metric}_mean' in t]
    assert scored, 'no scored trial to choose from'
    return sorted(scored, key=lambda t: (-t[f'val_{select_metric}_mean'], -t[f'val_{other}_mean'], t['trial_id']))[0]


def save_recipe(path, trial, retrain_seeds):
    '''recipe.json: the winner's knobs, best epoch and one threshold for every retrain seed (the mean of the
    tuning seeds' validation thresholds at the best epoch).'''
    mean_threshold = float(np.mean(list(trial['thresholds'].values())))
    frozen = {'operator': trial['operator'], 'features': trial['features'], 'recipe': trial['recipe'],
              'best_epoch': trial['best_epoch'], 'tuning_seeds': trial['seeds'], 'retrain_seeds': list(retrain_seeds),
              'thresholds': {str(s): mean_threshold for s in retrain_seeds}, 'tuning_thresholds': trial['thresholds'],
              'trial_id': trial['trial_id'], 'tag': trial['tag'],
              'val_pr_auc_mean': trial['val_pr_auc_mean'], 'val_pr_auc_std': trial['val_pr_auc_std'],
              'val_f1_mean': trial['val_f1_mean'], 'val_f1_std': trial['val_f1_std']}
    write_json_atomic(path, frozen)
    print(f'frozen recipe -> {path}: trial {trial["trial_id"]} ({trial["tag"]}), best epoch {trial["best_epoch"]}, '
          f'val PR-AUC {trial["val_pr_auc_mean"]:.4f} +- {trial["val_pr_auc_std"]:.4f}')
    return frozen


def load_recipe(path):
    assert os.path.exists(path), f'{path} not found: run aggregate.py after the search to freeze the winner'
    return json.load(open(path))


def load_trials(run_dir):
    return [json.load(open(p)) for p in sorted(glob.glob(f'{run_dir}/t*/trial.json'))]


def trials_table(trials, space, winner_id=None):
    '''One row per trial: the knobs, status, best epoch, the mean / std of every train and val metric.'''
    rows = []
    for t in trials:
        r = t['recipe']
        row = {'trial_id': t['trial_id'], 'tag': t['tag'], 'winner': t['trial_id'] == winner_id, 'status': t['status'],
               **{k: r[k] for k in space}, 'num_layers': r['num_layers'],
               'num_neighbors': '-'.join(str(n) for n in r['num_neighbors']), 'best_epoch': t.get('best_epoch'),
               'params': t['params'], 'invariant_params': t['invariant_params'], 'gap_pr_auc': t.get('gap_pr_auc')}
        for key in TRIAL_METRIC_KEYS:
            row[f'{key}_mean'], row[f'{key}_std'] = t.get(f'{key}_mean'), t.get(f'{key}_std')
        rows.append(row)
    return pd.DataFrame(rows)


def plot_val_curves(trials, winner_id, path, select_metric, max_labels=10):
    '''tuning_curves.png: the seed-mean validation curve of every scored trial (winner bold, top trials coloured).'''
    key = f'val_{select_metric}'
    plotted = sorted([t for t in trials if f'{key}_mean_curve' in t], key=lambda t: -t[f'{key}_mean'])
    fig, ax = plt.subplots(figsize=(12, 6))
    for rank, t in enumerate(plotted):
        curve, shown, is_winner = t[f'{key}_mean_curve'], rank < max_labels, t['trial_id'] == winner_id
        line, = ax.plot(range(1, len(curve) + 1), curve, linewidth=2.2 if is_winner else 1.0,
                        alpha=1.0 if shown else 0.25, color=None if shown else 'gray',
                        label=f"t{t['trial_id']:03d} {t['tag']} ({t[f'{key}_mean']:.3f} @ {t['best_epoch']})" if shown else None)
        ax.plot(t['best_epoch'], curve[t['best_epoch'] - 1], 'o', color=line.get_color())
    ax.set_xlabel('epoch')
    ax.set_ylabel(f'validation {select_metric} (seed mean)')
    ax.set_title(f'random search: seed-mean validation {select_metric} per trial (coloured: top {max_labels})')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- the checks of aggregate.py

def check_trial(trial, expected_recipe, trial_dir, seeds, epochs):
    '''Every check of one finished trial; an AssertionError names the trial.'''
    name = f"{trial['operator']} {trial['features']['mp']}/{trial['features']['readout']} t{trial['trial_id']:03d}"
    assert trial['recipe'] == expected_recipe, f'{name}: recipe differs from the drawn trial'
    for s in seeds:
        assert json.load(open(f'{trial_dir}/seed-{s}/done.json'))['test_split_loaded'] is False, f'{name} seed {s}: test loaded'
    if trial['status'] in ('oom', 'diverged') and 'best_epoch' not in trial:
        return
    diverged = trial['diverged_at'] or {}
    histories = {s: pd.read_csv(f'{trial_dir}/seed-{s}/history.csv') for s in seeds}
    for s, h in histories.items():
        expected = diverged[str(s)] - 1 if str(s) in diverged else epochs
        assert len(h) == expected, f'{name} seed {s}: {len(h)} epochs in history.csv, expected {expected}'
        assert (h['lr'] == trial['recipe']['lr']).all(), f'{name} seed {s}: the learning rate is not constant'
        assert not h[TRIAL_METRIC_KEYS + ['threshold']].isna().any().any(), f'{name} seed {s}: NaN in history.csv'
        row = h.loc[h['epoch'] == trial['best_epoch']].iloc[0]
        for key in TRIAL_METRIC_KEYS + ['threshold']:
            assert abs(trial['per_seed'][str(s)][key] - float(row[key])) < 1e-12, f'{name} seed {s}: trial.json {key} differs from history.csv'
    metric = trial['recipe']['select_metric']
    n_common = min(len(h) for h in histories.values())
    best = int(np.argmax(np.mean([h[f'val_{metric}'].to_numpy()[:n_common] for h in histories.values()], axis=0))) + 1
    assert best == trial['best_epoch'], f"{name}: best epoch {trial['best_epoch']} != {best} (argmax of the seed-mean val {metric})"


def check_final(results, history, frozen, seed):
    '''Every check of one finished final-retrain seed.'''
    name = f"{frozen['operator']} {frozen['features']['mp']}/{frozen['features']['readout']} seed {seed}"
    assert results['stop_epoch'] == frozen['best_epoch'] == len(history), f'{name}: epochs differ from recipe.json'
    assert results['threshold'] == frozen['thresholds'][str(seed)], f'{name}: threshold differs from recipe.json'
    assert results['recipe'] == frozen['recipe'], f'{name}: recipe differs from recipe.json'
    assert not [c for c in history if c.startswith('test')], f'{name}: test scored during training'
    assert (history['lr'] == frozen['recipe']['lr']).all(), f'{name}: the learning rate is not constant'
    assert all(np.isfinite(results[f'{split}_{m}']) for split in ('train', 'val', 'test') for m in METRICS), f'{name}: a metric is missing'
