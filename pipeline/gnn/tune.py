'''The GNN part of the search: a trial's recipe from its drawn values, and the checks aggregate.py runs.'''
import numpy as np
import pandas as pd

from pipeline.gnn.model import check_recipe
from pipeline.metrics import METRICS


def trial_recipe(base, values, neighbours):
    '''The recipe of one trial: the search file's values, the drawn shared values, the operator's own under conv,
    and one neighbour count per layer.'''
    shared = {k: v for k, v in values.items() if k in base}
    conv = {k: v for k, v in values.items() if k not in base}
    return check_recipe(dict(base, **shared, conv=conv, num_neighbors=list(neighbours[:values['num_layers']])))


def check_trial(trial, trial_dir, epochs):
    '''Every check of one full trial against its history.csv.'''
    history = pd.read_csv(f'{trial_dir}/history.csv')
    assert len(history) == epochs, f'{trial_dir}: {len(history)} epochs in history.csv, expected {epochs}'
    assert history['lr'].nunique() == 1, f'{trial_dir}: the learning rate is not constant'
    assert not history.isna().any().any(), f'{trial_dir}: NaN in history.csv'
    best = history.loc[history['val_pr_auc'].idxmax()]
    assert int(best['epoch']) == trial.user_attrs['best_epoch'], f'{trial_dir}: best epoch differs from history.csv'
    assert abs(best['val_pr_auc'] - trial.value) < 1e-12, f'{trial_dir}: score differs from history.csv'
    assert abs(best['threshold'] - trial.user_attrs['threshold']) < 1e-12, f'{trial_dir}: threshold differs from history.csv'


def check_final(results, history, frozen, seed):
    '''Every check of one finished final-retrain seed.'''
    name = f"{frozen['operator']} {frozen['features']['mp']}/{frozen['features']['readout']} seed {seed}"
    assert results['stop_epoch'] == frozen['best_epoch'] == len(history), f'{name}: epochs differ from recipe.json'
    assert results['threshold'] == frozen['threshold'], f'{name}: threshold differs from recipe.json'
    assert results['recipe'] == frozen['recipe'], f'{name}: recipe differs from recipe.json'
    assert not [c for c in history if c.startswith('test')], f'{name}: test scored during training'
    assert (history['lr'] == frozen['recipe']['lr']).all(), f'{name}: the learning rate is not constant'
    assert all(np.isfinite(results[f'{split}_{m}']) for split in ('train', 'val', 'test') for m in METRICS), f'{name}: a metric is missing'
