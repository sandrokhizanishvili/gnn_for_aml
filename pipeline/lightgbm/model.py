'''LightGBM on the 93 inputs the GNN readout sees (no message passing): one booster fit for a given number of
rounds, and its predictions.'''
import os

import lightgbm as lgb

LOSS_WEIGHT = 8.0          # laundering weight of the loss diagnostic in history.csv (= the GNN pos_weight)
THREADS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))


def fit_booster(params, settings, X, y, rounds, seed, val=None, callbacks=()):
    '''lgb.train with the settings + one configuration for `rounds` rounds, no early stopping. With val = (X, y),
    LightGBM scores the validation rows after every round with its own average precision (PR-AUC).'''
    dataset = lgb.Dataset(X, label=y)
    valid_sets = [lgb.Dataset(val[0], label=val[1], reference=dataset)] if val is not None else []
    lgb_params = dict(settings, **params, seed=seed, metric='average_precision', num_threads=THREADS)
    return lgb.train(lgb_params, dataset, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)


def predict_booster(booster, X, k=None):
    return booster.predict(X, num_iteration=k, num_threads=THREADS)
