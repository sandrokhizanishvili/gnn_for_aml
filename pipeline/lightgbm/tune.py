'''One LightGBM trial of the search.'''
import lightgbm as lgb
import numpy as np

from pipeline.lightgbm.model import fit_booster, predict_booster
from pipeline.metrics import best_f1_threshold


def run_trial(params, hp, X, y, rows, seed, report):
    '''Every round on the train rows; LightGBM records its validation PR-AUC after each round.
    Returns (the best round's PR-AUC, that round, its threshold and F1).'''
    X_val, y_val = X[rows['val']], y[rows['val']]
    curve = {}

    def report_curve(env):   # LightGBM calls this before it records the round: the curve holds the earlier rounds
        scores = curve.get('valid_0', {}).get('average_precision', [])
        if scores and len(scores) % hp['search']['report_every'] == 0:
            report(max(scores), len(scores))

    booster = fit_booster(params, hp['settings'], X[rows['train']], y[rows['train']], hp['rounds'], seed, (X_val, y_val),
                          [report_curve, lgb.record_evaluation(curve), lgb.log_evaluation(50)])
    scores = curve['valid_0']['average_precision']
    best_round = int(np.argmax(scores)) + 1
    threshold, val_f1 = best_f1_threshold(y_val, predict_booster(booster, X_val, best_round))
    return scores[best_round - 1], dict(best_epoch=best_round, threshold=threshold, val_f1=val_f1)
