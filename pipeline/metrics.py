'''Scoring, the same for every model: the validation threshold sweep, the seven metrics of a split, the per-epoch
history row, the threshold sensitivity table.'''
import time

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from pipeline.data import SPLITS

METRICS = ('f1', 'precision', 'recall', 'pr_auc', 'roc_auc', 'precision_at_5pct', 'recall_at_5pct')
METRIC_COLS = [f'{split}_{metric}' for split in SPLITS for metric in METRICS]
SENSITIVITY_DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)


def best_f1_threshold(y, p):
    '''The threshold that maximises F1 on the validation predictions -> (threshold, f1).'''
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[f1.argmax()]), float(f1.max())


def weighted_bce(y, p, pos_weight):
    '''The training objective from eval-mode probabilities: comparable train / val loss curves.'''
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p)))


def compute_metrics(y, p, threshold):
    '''The seven metrics of one split; the thresholded ones at the validation-chosen threshold.'''
    pred = (p >= threshold).astype(int)
    k = max(1, int(np.ceil(0.05 * len(p))))
    hits = int(y[np.argsort(-p, kind='stable')[:k]].sum())
    return {'f1': float(f1_score(y, pred)), 'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred)), 'pr_auc': float(average_precision_score(y, p)),
            'roc_auc': float(roc_auc_score(y, p)), 'precision_at_5pct': hits / k,
            'recall_at_5pct': hits / max(1, int(y.sum()))}


def epoch_row(epoch, y_train, p_train, y_val, p_val, pos_weight, t0):
    '''One history row: losses, F1 at the validation threshold, PR-AUC, ROC-AUC, threshold, seconds.'''
    threshold, val_f1 = best_f1_threshold(y_val, p_val)
    return {'epoch': epoch, 'train_loss': weighted_bce(y_train, p_train, pos_weight),
            'val_loss': weighted_bce(y_val, p_val, pos_weight),
            'train_f1': float(f1_score(y_train, (p_train >= threshold).astype(int))), 'val_f1': val_f1,
            'train_pr_auc': float(average_precision_score(y_train, p_train)),
            'val_pr_auc': float(average_precision_score(y_val, p_val)), 'val_roc_auc': float(roc_auc_score(y_val, p_val)),
            'threshold': threshold, 'seconds': time.time() - t0}


def score_splits(prob, y, rows, threshold):
    '''{split: metrics} and {split: {edge_id, y, prob}} from the probabilities of every row (tabular models).'''
    metrics = {s: compute_metrics(y[r], prob[r], threshold) for s, r in rows.items()}
    preds = {s: {'edge_id': np.arange(r.start, r.stop), 'y': y[r], 'prob': prob[r]} for s, r in rows.items()}
    return metrics, preds


def threshold_sensitivity(y, p, threshold):
    '''Test F1 / precision / recall at the frozen threshold +-0.05, +-0.10 and at 0.5: a robustness table, never used to choose.'''
    points = [('frozen', threshold + d, d) for d in SENSITIVITY_DELTAS] + [('paper 0.5', 0.5, 0.5 - threshold)]
    table = []
    for label, t, delta in points:
        t = float(min(max(t, 1e-6), 1 - 1e-6))
        pred = (p >= t).astype(int)
        table.append({'point': label, 'threshold': t, 'delta': float(delta), 'f1': float(f1_score(y, pred)),
                      'precision': float(precision_score(y, pred, zero_division=0)), 'recall': float(recall_score(y, pred))})
    return table
