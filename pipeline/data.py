'''Reading data/ (the output of prepare_data.py): the three cumulative snapshots, the edge-column sets, the node
encodings, the tuned GFP sheet, the train + val graph of the final retrain, the 93-column matrix of the tabular models.'''
import json
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from pipeline.gfp import normalise_gfp_sheet

SPLITS = ('train', 'val', 'test')
BASE_NODE_DIM = 6                          # the entity one-hots every run has; a node encoding comes after them
BASE_EDGE_DIM = 20                         # baseline edge columns, always edge_attr[:, :20]
GFP_WIDTH = {'default': 61, 'tuned': 64}   # default = the sheet inside the graph files; tuned = gfp_variants/tuned.npy
NODE_ENCODINGS = ('none', 'rwpe8', 'rwpe16')


def set_seed(seed):
    '''Seed python, numpy and torch: weights, neighbour samples and mini-batch order become reproducible.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_graphs(data_dir, gfp_sheet='default', splits=SPLITS):
    '''The snapshots and the column sets {none | base | gfp | full} -> (graphs, col_sets, node_dim).
    tuned replaces the GFP block by gfp_variants/tuned.npy, normalised on the train rows. Tuning passes splits
    ('train', 'val'): the test snapshot is never read.'''
    assert gfp_sheet == 'default' or tuple(splits) == SPLITS, 'the tuned GFP sheet is aligned with all three snapshots'
    graphs = {split: torch.load(f'{data_dir}/{split}_graph.pt', weights_only=False) for split in splits}
    assert graphs['train'].edge_attr.shape[1] == BASE_EDGE_DIM + GFP_WIDTH['default'], 'edge_attr must be 20 baseline + 61 GFP columns'

    if gfp_sheet == 'tuned':
        gfp_cols = json.load(open(f'{data_dir}/gfp_variants/tuned_cols.json'))
        sheet = np.load(f'{data_dir}/gfp_variants/tuned.npy')
        n_train, n_all = graphs['train'].edge_index.shape[1], graphs['test'].edge_index.shape[1]
        assert sheet.shape == (n_all, GFP_WIDTH['tuned']), f'tuned sheet {sheet.shape}: one row per edge of the test snapshot, 64 columns'
        sheet = torch.from_numpy(normalise_gfp_sheet(sheet, gfp_cols, n_train))
        for g in graphs.values():
            g.edge_attr = torch.cat([g.edge_attr[:, :BASE_EDGE_DIM], sheet[:g.edge_index.shape[1]]], dim=1)

    base_idx = list(range(BASE_EDGE_DIM))
    gfp_idx = list(range(BASE_EDGE_DIM, BASE_EDGE_DIM + GFP_WIDTH[gfp_sheet]))
    col_sets = {'none': [], 'base': base_idx, 'gfp': gfp_idx, 'full': base_idx + gfp_idx}
    print(f"edge_attr width {len(col_sets['full'])} (base {BASE_EDGE_DIM} + gfp {GFP_WIDTH[gfp_sheet]}, sheet {gfp_sheet})")
    for split, g in graphs.items():
        n_eval, n_pos = int(g.eval_mask.sum()), int((g.y[g.eval_mask] == 1).sum())
        print(f'{split:<5} nodes={g.num_nodes:,} edges={g.edge_index.shape[1]:,} '
              f'evaluated={n_eval:,} laundering={n_pos:,} ({n_pos / n_eval:.4%})')
    return graphs, col_sets, graphs['train'].x.shape[1]


def set_node_encoding(graphs, node_encoding, data_dir):
    '''graph.x <- [entity one-hots || rwpe<k> columns of that snapshot]; returns the node dim (6 or 6 + k).'''
    assert node_encoding in NODE_ENCODINGS, f'unknown node encoding {node_encoding!r}; choose from {NODE_ENCODINGS}'
    for split, graph in graphs.items():
        if node_encoding != 'none':
            k = int(node_encoding[len('rwpe'):])
            enc = torch.load(f'{data_dir}/rwpe/rwpe_k{k}_{split}.pt', weights_only=True).float()
            assert enc.shape == (graph.num_nodes, k), f'{split}: encoding {tuple(enc.shape)} != ({graph.num_nodes}, {k})'
            graph.x = torch.cat([graph.x, enc], dim=1)
    return graphs['train'].x.shape[1]


def make_trainval_graph(graphs):
    '''The final-retrain training graph: the val snapshot with every edge labelled and evaluated.'''
    train, val = graphs['train'], graphs['val']
    n_train, n_all = train.edge_index.shape[1], val.edge_index.shape[1]
    assert torch.equal(val.edge_index[:, :n_train], train.edge_index), 'val snapshot must start with the train edges'
    y = val.y.clone()
    y[:n_train] = train.y   # the val snapshot holds the train labels as -1
    assert bool(((y == 0) | (y == 1)).all()), 'every train+val edge must carry a 0/1 label'
    print(f'train+val graph: edges={n_all:,} (train {n_train:,} + val {n_all - n_train:,}) laundering={int((y == 1).sum()):,}')
    return Data(x=val.x, edge_index=val.edge_index, edge_attr=val.edge_attr, edge_time=val.edge_time, y=y,
                eval_mask=torch.ones(n_all, dtype=torch.bool), num_nodes=val.num_nodes)


def in_degree_histogram(graph):
    '''In-degree histogram of the graph trained on (PNA calibrates its scalers with it).'''
    return torch.bincount(degree(graph.edge_index[1], num_nodes=graph.num_nodes, dtype=torch.long))


def load_matrix(graphs):
    '''The tabular inputs: X [n_edges, 93] = edge_attr || x[src] || x[dst] of the largest snapshot loaded, y of every
    edge, rows {split: slice}. Snapshots are cumulative, so split s is the rows [n_{s-1}, n_s).'''
    splits = list(graphs)
    full = graphs[splits[-1]]
    src, dst = full.edge_index[0].numpy(), full.edge_index[1].numpy()
    x = full.x.numpy()
    X = np.hstack([full.edge_attr.numpy(), x[src], x[dst]]).astype(np.float32)
    y, rows, start = np.full(len(X), -1, dtype=np.int64), {}, 0
    for split in splits:
        g = graphs[split]
        stop = g.edge_index.shape[1]
        assert not g.eval_mask[:start].any() and g.eval_mask[start:].all(), f'{split}: evaluated edges are not its own window'
        y[start:stop] = g.y[start:stop].numpy()
        rows[split], start = slice(start, stop), stop
    assert (y >= 0).all() and not np.isnan(X).any()
    print(f'matrix {X.shape} | ' + ' | '.join(f'{s} rows {r.start:,}-{r.stop - 1:,} laundering {int(y[r].sum()):,}' for s, r in rows.items()))
    return X, y, rows
