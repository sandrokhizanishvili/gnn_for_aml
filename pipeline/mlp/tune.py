'''One MLP trial of the search.'''
from pipeline.gnn.model import count_params
from pipeline.mlp.train import fit_mlp


def run_trial(knobs, hp, X, y, rows, seed, report):
    '''Every epoch on the train rows -> (the best validation PR-AUC, its epoch, threshold and F1).'''
    model, history = fit_mlp(knobs, X, y, rows, rows['train'], hp['epochs'], seed, report)
    best = max(history, key=lambda row: row['val_pr_auc'])
    return best['val_pr_auc'], dict(best_epoch=best['epoch'], threshold=best['threshold'], val_f1=best['val_f1'],
                                    params=count_params(model))
