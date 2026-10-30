'''LightGBM on the 93 inputs the GNN readout sees (no message passing): one booster fit with early stopping, and
its predictions.'''
import os

import lightgbm as lgb
import numpy as np

LOSS_WEIGHT = 8.0          # laundering weight of the loss diagnostic in history.csv (= the GNN pos_weight)
THREADS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))


def fit_booster(params, hp, X_train, y_train, X_val, y_val, seed, callbacks=()):
    '''lgb.train with the settings + one configuration; early stopping on the validation PR-AUC (average precision,
    patience hp['early_stopping']); best_iteration = the best-PR-AUC round, the same rule as the other models.'''
    dataset = lgb.Dataset(X_train, label=y_train, params={'feature_pre_filter': False}, free_raw_data=False).construct()
    val = lgb.Dataset(X_val, label=y_val, reference=dataset, params={'feature_pre_filter': False}, free_raw_data=False).construct()
    lgb_params = dict(hp['settings'], **{k: v for k, v in params.items() if k != 'num_round'},
                      feature_pre_filter=False, seed=seed, metric='average_precision', num_threads=THREADS)
    return lgb.train(lgb_params, dataset, num_boost_round=params['num_round'], valid_sets=[val],
                     callbacks=list(callbacks) + [lgb.early_stopping(hp['early_stopping'], first_metric_only=True, verbose=False)])


def predict_booster(booster, X, k=None):
    return booster.predict(X, num_iteration=k, num_threads=THREADS).astype(np.float32)
