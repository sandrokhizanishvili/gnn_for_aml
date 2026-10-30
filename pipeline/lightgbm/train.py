'''LightGBM training at one configuration: the fit on the train rows, scored on train and val every
hp['check_every'] rounds for history.csv; the kept round is the early-stopping one.'''
import time

from pipeline.lightgbm.model import LOSS_WEIGHT, fit_booster, predict_booster
from pipeline.metrics import epoch_row


def fit_final_booster(params, hp, X, y, rows, seed):
    '''Returns (booster, history, kept round).'''
    X_train, y_train, X_val, y_val = X[rows['train']], y[rows['train']], X[rows['val']], y[rows['val']]
    marks = set(range(hp['check_every'], params['num_round'], hp['check_every'])) | {params['num_round']}
    history, last_time = [], time.time()

    def score(booster, k):
        nonlocal last_time
        row = epoch_row(k, y_train, predict_booster(booster, X_train, k), y_val, predict_booster(booster, X_val, k),
                        LOSS_WEIGHT, last_time)
        last_time = time.time()
        print(f"round {k:4d} | loss {row['train_loss']:.4f} / {row['val_loss']:.4f} | F1 {row['train_f1']:.4f} / "
              f"{row['val_f1']:.4f} @ {row['threshold']:.3f} | PR-AUC {row['train_pr_auc']:.4f} / {row['val_pr_auc']:.4f}")
        return row

    def checkpoint(env):
        if env.iteration + 1 in marks:
            history.append(score(env.model, env.iteration + 1))

    booster = fit_booster(params, hp, X_train, y_train, X_val, y_val, seed, callbacks=[checkpoint])
    kept = int(booster.best_iteration)
    if kept not in {h['epoch'] for h in history}:
        history.append(score(booster, kept))
        history.sort(key=lambda h: h['epoch'])
    return booster, history, kept
