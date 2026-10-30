'''Data preparation: the raw IBM HI-Small files in data/raw -> every input of run.py (conf/prepare_data.yaml).

    python prepare_data.py               graphs (features, split, the three snapshots) -> rwpe -> gfp_tuned
    python prepare_data.py step=rwpe     one step; each step skips files that already exist

Writes under paths.data: train/val/test_graph.pt, feature_meta.json, rwpe/rwpe_k<k>_<split>.pt,
gfp_variants/tuned.npy + tuned_cols.json. Same operations and order as the thesis notebooks, so the files come out
identical.'''
import json
import os
import subprocess

import hydra
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from pipeline.gfp import normalise_gfp_sheet, stream

RAW_COLS = ['Timestamp', 'src_bank', 'src_account', 'dst_bank', 'dst_account', 'Amount Received', 'Receiving Currency',
            'Amount Paid', 'Payment Currency', 'Payment Format', 'label']
SPLITS = ('train', 'val', 'test')
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- raw file, edge and node features

def load_raw(cfg):
    '''Transactions before the cutoff, time-sorted, with Amount_USD; the account -> node index map (first appearance).'''
    edges = pd.read_csv(f"{cfg['raw']}/HI-Small_Trans.csv", names=RAW_COLS, header=0, low_memory=False)
    edges['Timestamp'] = pd.to_datetime(edges['Timestamp'])
    edges = edges[edges['Timestamp'] < cfg['cutoff']].sort_values('Timestamp', kind='stable').reset_index(drop=True)
    fx = pd.read_csv(os.path.join(HERE, cfg['fx_rates']), float_precision='round_trip')   # the default parser is 1 bit off
    fx = fx.set_index(['currency', 'date'])['rate']
    key = pd.MultiIndex.from_arrays([edges['Payment Currency'], edges['Timestamp'].dt.strftime('%Y-%m-%d')])
    edges['Amount_USD'] = edges['Amount Paid'] * fx.reindex(key).fillna(1.0).to_numpy()   # USD rows: 1.0
    edges = edges.drop(columns=['Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency'])
    account_to_idx = {a: i for i, a in enumerate(pd.concat([edges['src_account'], edges['dst_account']]).unique())}
    print(f"{len(edges):,} transactions before {cfg['cutoff']} | laundering {int(edges['label'].sum()):,} | accounts {len(account_to_idx):,}")
    return edges, account_to_idx


def baseline_features(edges, t1, m):
    '''The 20 baseline edge columns: amount, structuring band, time encodings, flags, time since the account's previous
    transaction, bank risk (train-label target encoding, causal on the train rows).'''
    f = pd.DataFrame(index=edges.index)
    f['Amount_Log'] = np.log1p(edges['Amount_USD'])
    f['Struct_Band'] = edges['Amount_USD'].between(9000, 10000).astype(np.int8)
    hour, dow = edges['Timestamp'].dt.hour, edges['Timestamp'].dt.dayofweek
    f['Hour_Sin'], f['Hour_Cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f['DayOfWeek_Sin'], f['DayOfWeek_Cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    f['Is_Weekend'] = (dow >= 5).astype(np.int8)
    f['Is_Self_Loop'] = (edges['src_account'] == edges['dst_account']).astype(np.int8)
    f['Same_Bank'] = (edges['src_bank'] == edges['dst_bank']).astype(np.int8)
    seconds, ten_days = edges['Timestamp'].astype('int64') // 10**9, 10 * 24 * 3600
    f['Dt_Src_Log'] = np.log1p((seconds - seconds.groupby(edges['src_account']).shift()).fillna(ten_days))
    f['Dt_Dst_Log'] = np.log1p((seconds - seconds.groupby(edges['dst_account']).shift()).fillna(ten_days))
    is_train, label = np.arange(len(edges)) < t1, edges['label']
    prior = label[is_train].mean()
    for col, bank_col in (('Src_Bank_Risk', 'src_bank'), ('Dst_Bank_Risk', 'dst_bank')):
        bank = edges[bank_col]
        causal = (label.groupby(bank).cumsum() - label + m * prior) / (label.groupby(bank).cumcount() + m)   # strictly earlier rows
        stats = edges[is_train].groupby(bank_col)['label'].agg(['sum', 'count'])
        frozen = bank.map((stats['sum'] + m * prior) / (stats['count'] + m)).fillna(prior)                   # the train window, frozen
        f[col] = np.where(is_train, causal, frozen).astype(np.float32)
    fmt = pd.get_dummies(edges['Payment Format'], prefix='PayFmt').astype(np.int8)
    return pd.concat([f, fmt], axis=1)


def node_features(cfg, account_to_idx):
    '''One row per node (account_to_idx order): the one-hot of the entity type ("Corporation #33520" -> Corporation).'''
    accounts = pd.read_csv(f"{cfg['raw']}/HI-Small_accounts.csv", low_memory=False).drop_duplicates('Account Number')
    onehot = pd.get_dummies(accounts['Entity Name'].str.extract(r'^([A-Za-z]+)', expand=False), prefix='EntityType').astype(np.int8)
    onehot.index = accounts['Account Number']
    x = onehot.reindex(account_to_idx.keys()).fillna(0).to_numpy(dtype=np.float32)   # accounts missing from the file -> 0
    return torch.tensor(x), list(onehot.columns)


# ---------------------------------------------------------------- GFP

def gfp_colnames(p):
    '''The column names snapml emits for a parameter set, in its order (pattern histograms, then vertex statistics).'''
    names, cols = ['fan', 'deg', 'ratio', 'avg', 'sum', 'min', 'max', 'median', 'var', 'skew', 'kurtosis'], []
    for pattern in ('fan', 'degree', 'scatter-gather', 'temp-cycle', 'lc-cycle'):
        if p[pattern]:
            bins = p[f'{pattern}_bins']
            edges_ = [f'{lo}-{hi}' for lo, hi in zip(bins, bins[1:])] + [f'{bins[-1]}-inf']
            sides = ['in', 'out'] if pattern in ('fan', 'degree') else [None]
            cols += [f'{pattern}_{s}_bins_{e}' if s else f'{pattern}_bins_{e}' for s in sides for e in edges_]
    for who in ('source', 'dest'):
        for direction in ('out', 'in'):
            cols += [f'{who}_{names[k]}_{direction}' for k in p['vertex_stats_feats'] if k <= 2]
            for label in ('ts', 'amt'):
                cols += [f'{who}_{names[k]}_{label}_{direction}' for k in p['vertex_stats_feats'] if k >= 3]
    return cols


def run_gfp(cfg, edges, account_to_idx, params, work_dir):
    '''snapml GFP over every transaction, streamed in batches -> float32 [n, features] (the 5 input columns dropped).'''
    seconds = edges['Timestamp'].astype('int64') // 10**9
    X = np.column_stack([np.arange(len(edges)), edges['src_account'].map(account_to_idx), edges['dst_account'].map(account_to_idx),
                         seconds - seconds.min(), edges['Amount_USD']]).astype(np.float64)
    if not cfg['gfp']['python']:
        return stream(X, params, cfg['gfp']['batch'])[:, 5:]
    paths = [os.path.abspath(f'{work_dir}/_gfp_{name}') for name in ('input.npy', 'params.json', 'output.npy')]
    np.save(paths[0], X)
    json.dump(params, open(paths[1], 'w'))
    script = os.path.join(HERE, 'pipeline', 'gfp.py')
    wsl = cfg['gfp']['python'].startswith('wsl')
    args = [('/mnt/' + p[0].lower() + p[2:].replace('\\', '/')) if wsl else p for p in [script] + paths]
    print(f"gfp: {len(X):,} edges in batches of {cfg['gfp']['batch']} through {cfg['gfp']['python']} ...", flush=True)
    subprocess.run(cfg['gfp']['python'].split() + args + [str(cfg['gfp']['batch'])], check=True)
    out = np.load(paths[2])[:, 5:]
    for p in paths:
        os.remove(p)
    return out


# ---------------------------------------------------------------- the three steps

def step_graphs(cfg, out):
    '''Features, the temporal split and the three cumulative snapshots.'''
    edges, account_to_idx = load_raw(cfg)
    t1, t2 = (int(len(edges) * f) for f in cfg['split'])
    base = baseline_features(edges, t1, cfg['bank_risk_m'])
    gfp_cols = gfp_colnames(cfg['gfp']['default'])
    gfp = pd.DataFrame(run_gfp(cfg, edges, account_to_idx, cfg['gfp']['default'], out), columns=gfp_cols)
    first = (edges.groupby('src_account').cumcount() == 0).to_numpy()   # causality: a first transaction sees at most its batch
    assert gfp.loc[first, 'source_deg_out'].max() < cfg['gfp']['batch'], 'degrees beyond one batch: future leakage'
    feats = pd.concat([base, gfp], axis=1).fillna(0).astype(np.float32)
    feats[gfp_cols] = normalise_gfp_sheet(feats[gfp_cols].to_numpy(), gfp_cols, t1)   # fit on the train rows only
    x, node_cols = node_features(cfg, account_to_idx)
    edge_index = torch.tensor(np.stack([edges['src_account'].map(account_to_idx).to_numpy(), edges['dst_account'].map(account_to_idx).to_numpy()]))
    edge_time = torch.tensor(edges['Timestamp'].astype('int64').to_numpy() // 10**9)
    edge_attr, labels = torch.tensor(feats.to_numpy(dtype=np.float32)), edges['label'].to_numpy()
    for split, n_rows, eval_start in (('train', t1, 0), ('val', t2, t1), ('test', len(edges), t2)):
        eval_mask = np.arange(n_rows) >= eval_start
        # .clone(): a view would make torch.save write the whole array into every snapshot file
        graph = Data(x=x, edge_index=edge_index[:, :n_rows].clone(), edge_attr=edge_attr[:n_rows].clone(),
                     edge_time=edge_time[:n_rows].clone(), y=torch.tensor(np.where(eval_mask, labels[:n_rows], -1)),
                     eval_mask=torch.tensor(eval_mask), num_nodes=len(account_to_idx))
        torch.save(graph, f'{out}/{split}_graph.pt')
        print(f'{split:<5} edges={n_rows:>9,} evaluated={int(eval_mask.sum()):>9,} laundering={int((graph.y == 1).sum()):,}')
    base_cols = list(base.columns)
    json.dump({'BASE_EDGE_COLS': base_cols, 'GFP_FEAT_COLS': gfp_cols, 'EDGE_FEAT_COLS': base_cols + gfp_cols,
               'NODE_FEAT_COLS': node_cols, 'EDGE_DIM': len(base_cols) + len(gfp_cols), 'NODE_DIM': len(node_cols),
               'truncation': f"kept Timestamp < {cfg['cutoff']}", 'bank_risk_smoothing_m': cfg['bank_risk_m'],
               'split_rows': {'train': t1, 'val': t2, 'test': len(edges)}}, open(f'{out}/feature_meta.json', 'w'), indent=2)


def rwpe(src, dst, num_nodes, k_max):
    '''Random-walk encoding: RWPE(v)[k-1] = P^k[v, v] with P = D_out^-1 A from the edges without self-loops.'''
    keep = src != dst
    src, dst = src[keep], dst[keep]
    adjacency = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(num_nodes, num_nodes))
    adjacency.sum_duplicates()
    inv_degree = (1.0 / np.maximum(np.bincount(src, minlength=num_nodes), 1)).astype(np.float32)
    transition = (sp.diags(inv_degree) @ adjacency).tocsr().astype(np.float32)
    out, power = np.zeros((num_nodes, k_max), dtype=np.float32), transition.copy()
    out[:, 0] = power.diagonal()
    for k in range(2, k_max + 1):
        power = (power @ transition).tocsr()
        out[:, k - 1] = power.diagonal()
        print(f'  step {k:>2} | nnz(P^{k}) = {power.nnz:,}', flush=True)
    return out


def step_rwpe(cfg, out):
    '''One encoding file per (k, split) from the snapshot's own edges.'''
    os.makedirs(f'{out}/rwpe', exist_ok=True)
    graph = torch.load(f'{out}/test_graph.pt', weights_only=False)   # every snapshot is a prefix of the test edges
    n_rows = json.load(open(f'{out}/feature_meta.json'))['split_rows']
    src, dst = graph.edge_index.numpy()
    for k in cfg['rwpe']['k']:
        for split in SPLITS:
            path = f'{out}/rwpe/rwpe_k{k}_{split}.pt'
            if os.path.exists(path):
                continue
            print(f'rwpe k={k} {split}: {n_rows[split]:,} edges')
            enc = rwpe(src[:n_rows[split]], dst[:n_rows[split]], graph.num_nodes, k)
            assert not np.isnan(enc).any() and enc.min() >= 0 and enc.max() <= 1
            torch.save(torch.from_numpy(enc), path)


def step_gfp_tuned(cfg, out):
    '''The 64-column GFP sheet with the longer windows, raw (run.py normalises it on the train rows at load time).'''
    os.makedirs(f'{out}/gfp_variants', exist_ok=True)
    edges, account_to_idx = load_raw(cfg)
    # The thesis built this sheet from the saved feature file, so its amounts had been through float32 and CSV text:
    # the same round trip here keeps the sheet identical (it moves large amounts by up to ~1e-6 relative).
    edges['Amount_USD'] = np.expm1(np.log1p(edges['Amount_USD'].to_numpy()).astype(np.float32).astype(str).astype(np.float64))
    params = dict(cfg['gfp']['default'], **cfg['gfp']['tuned'])
    cols = gfp_colnames(params)
    sheet = np.ascontiguousarray(run_gfp(cfg, edges, account_to_idx, params, out), dtype=np.float32)
    assert sheet.shape == (len(edges), len(cols)) and not np.isnan(sheet).any()
    np.save(f'{out}/gfp_variants/tuned.npy', sheet)
    json.dump(cols, open(f'{out}/gfp_variants/tuned_cols.json', 'w'), indent=1)
    print(f'tuned sheet {sheet.shape} -> {out}/gfp_variants/')


@hydra.main(config_path='conf', config_name='prepare_data', version_base=None)
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    out = cfg['paths']['data']
    assert os.path.isdir(cfg['raw']), f"{cfg['raw']}: put HI-Small_Trans.csv and HI-Small_accounts.csv there (or set raw=<folder>)"
    os.makedirs(out, exist_ok=True)
    steps = ('graphs', 'rwpe', 'gfp_tuned') if cfg['step'] == 'all' else (cfg['step'],)
    done = {'graphs': f'{out}/test_graph.pt', 'rwpe': f"{out}/rwpe/rwpe_k{max(cfg['rwpe']['k'])}_test.pt",
            'gfp_tuned': f'{out}/gfp_variants/tuned.npy'}
    for step in steps:
        if os.path.exists(done[step]):
            print(f'{step}: {done[step]} exists, skipping')
            continue
        print(f'=== {step} -> {out}', flush=True)
        {'graphs': step_graphs, 'rwpe': step_rwpe, 'gfp_tuned': step_gfp_tuned}[step](cfg, out)


if __name__ == '__main__':
    main()
