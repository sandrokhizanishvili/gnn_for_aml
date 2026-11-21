'''The LightGBM final refit: the frozen configuration on the train + val rows for the frozen number of rounds.'''
import time

from pipeline.lightgbm.model import LOSS_WEIGHT, fit_booster, predict_booster
from pipeline.metrics import epoch_row


def fit_final_booster(params, hp, X, y, rows, rounds, seed):
    '''Train and val are scored every hp['check_every'] rounds for history.csv only (val is training data here).
    Returns (booster, history).'''
    fit_rows = slice(rows['train'].start, rows['val'].stop)
    X_train, y_train, X_val, y_val = X[rows['train']], y[rows['train']], X[rows['val']], y[rows['val']]
    marks = set(range(hp['check_every'], rounds, hp['check_every'])) | {rounds}
    history, last_time = [], time.time()

    def checkpoint(env):
        nonlocal last_time
        k = env.iteration + 1
        if k in marks:
            row = epoch_row(k, y_train, predict_booster(env.model, X_train, k), y_val, predict_booster(env.model, X_val, k),
                            LOSS_WEIGHT, last_time)
            print(f"round {k:4d} | loss {row['train_loss']:.4f} / {row['val_loss']:.4f} | F1 {row['train_f1']:.4f} / "
                  f"{row['val_f1']:.4f} @ {row['threshold']:.3f} | PR-AUC {row['train_pr_auc']:.4f} / {row['val_pr_auc']:.4f}", flush=True)
            history.append(row)
            last_time = time.time()

    booster = fit_booster(params, hp['settings'], X[fit_rows], y[fit_rows], rounds, seed, callbacks=[checkpoint])
    return booster, history
