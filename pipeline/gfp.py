'''GFP = the graph-pattern features of IBM snapml (fan-in / fan-out, scatter-gather, cycles, vertex statistics).
stream() runs the preprocessor over a time-sorted edge list in small batches, so an edge sees only earlier batches
and its own (one transform over everything would let every edge see the future); normalise_gfp_sheet() is the
train-fit normalisation of the data preparation, also applied to the tuned sheet at load time.

    python pipeline/gfp.py <input.npy> <params.json> <output.npy> <batch>      (numpy + snapml only; WSL on Windows)
input float64 [n, 5] = txn_id, src_idx, dst_idx, seconds, amount_usd;  output float32 [n, 5 + features]'''
import json
import sys
import time

import numpy as np


def stream(X, params, batch):
    '''snapml transform in causal batches -> float32 [n, 5 + features].'''
    from snapml import GraphFeaturePreprocessor
    gfp = GraphFeaturePreprocessor()
    gfp.set_params(params)
    out, t0 = None, time.time()
    for start in range(0, len(X), batch):
        part = gfp.transform(X[start:start + batch])
        if out is None:
            out = np.empty((len(X), part.shape[1]), dtype=np.float32)
        out[start:start + len(part)] = part
        if start % 1_000_000 < batch:
            print(f'  gfp {start:,}/{len(X):,} ({time.time() - t0:.0f}s)', flush=True)
    return out


def normalise_gfp_sheet(values, cols, n_train):
    '''Fit on the first n_train rows only. Pattern bins untouched; fan / deg / sum / avg / var / kurtosis: log1p ->
    clip at train p1 / p99 -> StandardScaler; ratio / skew: clip -> StandardScaler.'''
    from sklearn.preprocessing import StandardScaler   # here so the snapml CLI above needs no scikit-learn
    assert not np.isnan(values).any(), 'GFP sheet contains NaN'
    pattern_cols = [c for c in cols if '_bins_' in c]
    vertex_cols = [c for c in cols if c not in pattern_cols]
    log_cols = [c for c in vertex_cols if any(s in c for s in ['fan', 'deg', 'sum', 'avg', 'var', 'kurtosis'])]
    std_cols = [c for c in vertex_cols if c not in log_cols]
    out = values.copy()
    for group, use_log in [(log_cols, True), (std_cols, False)]:
        if not group:
            continue
        idx = [cols.index(c) for c in group]
        vals = values[:, idx]
        if use_log:
            vals = np.log1p(np.clip(vals, 0, None))
        p01 = np.percentile(vals[:n_train], 1, axis=0)
        p99 = np.percentile(vals[:n_train], 99, axis=0)
        vals = np.clip(vals, p01, p99)
        scaler = StandardScaler().fit(vals[:n_train])
        out[:, idx] = scaler.transform(vals).astype(np.float32)
    return out


if __name__ == '__main__':
    np.save(sys.argv[3], stream(np.load(sys.argv[1]), json.load(open(sys.argv[2])), int(sys.argv[4])))
