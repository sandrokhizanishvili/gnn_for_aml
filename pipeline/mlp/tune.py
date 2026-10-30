'''The MLP search: unique uniform draws from the yaml space (trial 0 = the hand-picked start), every trial trained
once per search seed, ranked by the mean best validation PR-AUC; resumes from trials.csv.'''
import json
import time

import numpy as np

from pipeline.gnn.model import count_params
from pipeline.mlp.train import fit_mlp
from pipeline.results import check_draw, log_trial, read_trials


def draw_trials(search):
    '''{trial: knobs}: trial 0 = search['start'], then unique uniform draws over search['space'].'''
    rng, configs = np.random.default_rng(search['draw_seed']), {0: dict(search['start'])}
    seen = {json.dumps(configs[0], sort_keys=True)}
    while len(configs) < search['n_trials']:
        knobs = {name: values[rng.integers(len(values))] for name, values in search['space'].items()}
        key = json.dumps(knobs, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs[len(configs)] = knobs
    return configs


def random_search(X, y, rows, hp, out_dir):
    '''Returns the winner.'''
    s, configs = hp['search'], draw_trials(hp['search'])
    check_draw(f'{out_dir}/configs.json', configs)
    trials = read_trials(f'{out_dir}/trials.csv')
    print(f"random search: {s['n_trials']} trials x seeds {s['seeds']} | {len(trials)} fits logged")
    for trial, knobs in configs.items():
        for seed in s['seeds']:
            if len(trials) and ((trials['trial'] == trial) & (trials['seed'] == seed)).any():
                continue
            print(f'--- trial {trial} seed {seed}')
            t0 = time.time()
            model, history, best_epoch = fit_mlp(knobs, hp, X, y, rows, seed)
            best = history[best_epoch - 1]
            trials = log_trial(f'{out_dir}/trials.csv', trials, {
                'trial': trial, 'seed': seed, 'params': count_params(model), 'best_epoch': best_epoch, 'epochs_trained': len(history),
                'val_pr_auc': best['val_pr_auc'], 'val_f1': best['val_f1'], 'threshold': best['threshold'],
                'seconds': time.time() - t0, 'knobs': json.dumps(knobs)})
    scores = trials.groupby('trial').agg(val_pr_auc=('val_pr_auc', 'mean'), val_f1=('val_f1', 'mean'), n=('seed', 'count')).reset_index()
    ranked = scores[scores['n'] == len(s['seeds'])].sort_values(['val_pr_auc', 'val_f1', 'trial'], ascending=[False, False, True])
    winner = int(ranked['trial'].iloc[0])
    return {'trial': winner, 'params': configs[winner], 'val_pr_auc': float(ranked['val_pr_auc'].iloc[0]),
            'val_f1': float(ranked['val_f1'].iloc[0])}
