'''One MLP trial of the search.'''
import os

from pipeline.gnn.model import count_params
from pipeline.metrics import best_epoch_scores
from pipeline.mlp.train import fit_mlp


def run_trial(knobs, hp, X, y, rows, seed, report, trial_dir):
    '''Every epoch on the train rows -> (the best validation PR-AUC, its epoch, threshold and train / val metrics).'''
    os.makedirs(trial_dir, exist_ok=True)
    model, history = fit_mlp(knobs, X, y, rows, rows['train'], hp['epochs'], seed, trial_dir, report)
    best = best_epoch_scores(history)
    return best['val_pr_auc'], dict(best, params=count_params(model))
