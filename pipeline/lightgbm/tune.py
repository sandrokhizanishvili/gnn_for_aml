'''The LightGBM search: successive halving (Altman et al. 2023 Appendix C) over random configurations drawn from
the yaml space; resumes from trials.csv.'''
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score

from pipeline.lightgbm.model import fit_booster, predict_booster
from pipeline.metrics import best_f1_threshold
from pipeline.results import check_draw, log_trial, read_trials


def sample_params(rng, space):
    '''One configuration: ints uniform, floats uniform, log ranges log-uniform (space = {name: [kind, low, high]}).'''
    out = {}
    for name, (kind, low, high) in space.items():
        if kind == 'int':
            out[name] = int(rng.integers(low, high + 1))
        elif kind == 'log':
            out[name] = float(10 ** rng.uniform(np.log10(low), np.log10(high)))
        else:
            out[name] = float(rng.uniform(low, high))
    return out


def halving_schedule(x0, eta, r0):
    '''[(configs alive, train fraction)]: x0 @ r0, x0/eta @ r0*eta, ... until the fraction reaches 1.'''
    rounds, n, f = [], x0, r0
    while True:
        rounds.append((n, min(f, 1.0)))
        if f >= 1.0 or n // eta < 1:
            return rounds
        n, f = n // eta, f * eta


def successive_halving(X, y, rows, hp, out_dir):
    '''Each round keeps the best 1/eta configurations on more (the most recent) train rows, scored on the full
    validation set at the kept round; ranked by search['select']. Returns the winner.'''
    s = hp['search']
    rng = np.random.default_rng(s['draw_seed'])
    configs = {t: sample_params(rng, s['space']) for t in range(s['n_configs'])}
    check_draw(f'{out_dir}/configs.json', configs)
    trials = read_trials(f'{out_dir}/trials.csv')
    X_train, y_train, X_val, y_val = X[rows['train']], y[rows['train']], X[rows['val']], y[rows['val']]
    schedule = halving_schedule(s['n_configs'], s['eta'], s['first_fraction'])
    print(f'successive halving {schedule} | ranked by {s["select"]} | {len(trials)} trial rows logged')
    alive = list(range(s['n_configs']))
    for round_no, (n_alive, fraction) in enumerate(schedule, 1):
        alive = alive[:n_alive]
        n_rows = int(round(fraction * len(y_train)))
        start = len(y_train) - n_rows
        for trial in alive:
            if len(trials) and ((trials['trial'] == trial) & (trials['round'] == round_no)).any():
                continue
            t0 = time.time()
            booster = fit_booster(configs[trial], hp, X_train[start:], y_train[start:], X_val, y_val, s['draw_seed'])
            p_val = predict_booster(booster, X_val, booster.best_iteration)
            threshold, val_f1 = best_f1_threshold(y_val, p_val)
            row = {'trial': trial, 'round': round_no, 'fraction': fraction, 'n_rows': n_rows, 'val_f1': val_f1,
                   'threshold': threshold, 'val_pr_auc': float(average_precision_score(y_val, p_val)),
                   'rounds': int(booster.best_iteration), 'seconds': time.time() - t0, 'params': json.dumps(configs[trial])}
            trials = log_trial(f'{out_dir}/trials.csv', trials, row)
            print(f"round {round_no} trial {trial:3d} | val F1 {val_f1:.4f} @ {threshold:.3f} | PR-AUC {row['val_pr_auc']:.4f} "
                  f"| rounds {row['rounds']} | {row['seconds']:.0f}s")
        this_round = trials[(trials['round'] == round_no) & trials['trial'].isin(alive)]
        alive = this_round.sort_values([s['select'], 'trial'], ascending=[False, True])['trial'].astype(int).tolist()
    best = trials[(trials['trial'] == alive[0]) & (trials['round'] == len(schedule))].iloc[0]
    return {'trial': int(alive[0]), 'params': configs[alive[0]], 'val_pr_auc': float(best['val_pr_auc']),
            'val_f1': float(best['val_f1']), 'threshold': float(best['threshold']), 'rounds': int(best['rounds'])}
