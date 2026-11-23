'''One LightGBM trial of the search.'''
import lightgbm as lgb
import numpy as np
import optuna

from pipeline.lightgbm.model import fit_booster, predict_booster
from pipeline.metrics import best_f1_threshold, train_val_metrics


def run_trial(params, hp, X, y, rows, seed, report):
    '''Every round on the train rows; LightGBM records its validation PR-AUC after each round. Returns (the best
    round's PR-AUC, that round, its threshold and train / val metrics); the PR-AUC is None when the pruner stopped
    the trial.'''
    X_train, y_train, X_val, y_val = X[rows['train']], y[rows['train']], X[rows['val']], y[rows['val']]
    curve, kept = {}, {}

    def report_curve(env):   # LightGBM calls this before it records the round: the curve holds the earlier rounds
        kept['booster'] = env.model
        scores = curve.get('valid_0', {}).get('average_precision', [])
        if scores and len(scores) % hp['search']['report_every'] == 0:
            report(max(scores), len(scores))

    try:
        booster, pruned = fit_booster(params, hp['settings'], X_train, y_train, hp['rounds'], seed, (X_val, y_val),
                                      [report_curve, lgb.record_evaluation(curve), lgb.log_evaluation(50)]), False
    except optuna.TrialPruned:   # the booster so far gets its metrics like a full trial
        booster, pruned = kept['booster'], True
    scores = curve['valid_0']['average_precision']
    best_round = int(np.argmax(scores)) + 1
    p_train, p_val = predict_booster(booster, X_train, best_round), predict_booster(booster, X_val, best_round)
    threshold, val_f1 = best_f1_threshold(y_val, p_val)
    info = dict(train_val_metrics(y_train, p_train, y_val, p_val, threshold), best_epoch=best_round, threshold=threshold,
                val_f1=val_f1)
    return (None if pruned else scores[best_round - 1]), info
