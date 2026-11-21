'''The GNN: one fixed template (node projection -> message-passing layers -> readout on the seed edge) in which only
the graph convolution differs between the operators. The recipe dict (conf/hyperparameters/gnn/) sets every
size and knob; identical module names across recipes keep every checkpoint loadable.'''
import re

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GINConv, GINEConv, GPSConv, PNAConv, TransformerConv

from pipeline.data import BASE_NODE_DIM

OPERATORS = ('gin', 'pna', 'gat', 'transformer', 'gps')
HEADS = 4                                    # attention operators: HEADS heads x (hidden / HEADS) = hidden
CONV = {                                     # each operator's own settings in the fixed architecture
    'gin': dict(train_eps=True, mlp_layers=2, mlp_width=1),
    'pna': dict(degree_scalers=True, towers=1, divide_input=False),
    'gat': dict(heads=HEADS, share_weights=False, attention_dropout=0.0),
    'transformer': dict(heads=HEADS, beta=False, attention_dropout=0.0),
    'gps': {},
}
RECIPE_KEYS = ('hidden', 'num_layers', 'dropout', 'norm', 'residual', 'ffn', 'activation', 'conv', 'num_neighbors',
               'batch_size', 'epochs', 'lr', 'weight_decay', 'pos_weight', 'self_loops', 'readout_layers',
               'readout_width', 'lr_schedule', 'num_workers', 'select_metric')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_recipe(r):
    '''Every knob present and consistent.'''
    assert set(r) == set(RECIPE_KEYS), f'recipe keys {sorted(set(r) ^ set(RECIPE_KEYS))} missing or unknown'
    assert len(r['num_neighbors']) == r['num_layers'], 'one neighbour count per message-passing layer'
    for parts in (HEADS, r['conv'].get('heads', 1), r['conv'].get('towers', 1)):
        assert r['hidden'] % parts == 0, 'hidden must split into the attention heads / the towers'
    return r


def gin_mlp(hidden, layers, width):
    '''The GIN node MLP: Linear -> BatchNorm -> ReLU -> Linear, width x hidden wide inside; one Linear when layers = 1.'''
    if layers == 1:
        return nn.Sequential(nn.Linear(hidden, hidden))
    inner = width * hidden
    return nn.Sequential(nn.Linear(hidden, inner), nn.BatchNorm1d(inner), nn.ReLU(), nn.Linear(inner, hidden))


def make_conv(operator, hidden, mp_edge_dim, deg_hist=None, self_loops=True, local='gin', conv=None):
    '''One graph convolution: the only place the operators differ. mp_edge_dim = 0 keeps edge features out.
    conv = the settings a search changed, on top of CONV[operator].'''
    edge = dict(edge_dim=mp_edge_dim) if mp_edge_dim > 0 else {}
    c = dict(CONV[operator], **(conv or {}))
    if operator == 'gin':
        mlp = gin_mlp(hidden, c['mlp_layers'], c['mlp_width'])
        return GINEConv(mlp, train_eps=c['train_eps'], **edge) if edge else GINConv(mlp, train_eps=c['train_eps'])
    if operator == 'pna':
        scalers = ['identity', 'amplification', 'attenuation'] if c['degree_scalers'] else ['identity']
        return PNAConv(hidden, hidden, aggregators=['mean', 'max', 'min', 'std'], scalers=scalers, deg=deg_hist,
                       towers=c['towers'], divide_input=c['divide_input'], **edge)
    if operator == 'gat':
        return GATv2Conv(hidden, hidden // c['heads'], heads=c['heads'], add_self_loops=self_loops, fill_value='mean',
                         bias=True, share_weights=c['share_weights'], dropout=c['attention_dropout'], **edge)
    if operator == 'transformer':
        return TransformerConv(hidden, hidden // c['heads'], heads=c['heads'], concat=True, beta=c['beta'],
                               dropout=c['attention_dropout'], root_weight=True, **edge)
    if operator == 'gps':                    # local operator on the edges + attention over every node of the subgraph
        return GPSConv(hidden, make_conv(local, hidden, mp_edge_dim, deg_hist), heads=HEADS, norm=None,
                       attn_type='performer', attn_kwargs=dict(head_channels=hidden // HEADS))
    raise ValueError(f'unknown operator {operator!r}; choose from {OPERATORS}')


def make_norm(norm, hidden):
    return {'none': nn.Identity(), 'batch': nn.BatchNorm1d(hidden), 'layer': nn.LayerNorm(hidden)}[norm]


def make_activation(name):
    '''relu | prelu (one learnable slope) | swish.'''
    return {'relu': nn.ReLU, 'prelu': nn.PReLU, 'swish': nn.SiLU}[name]()


class MPLayer(nn.Module):
    '''One message-passing step: h <- h + Dropout(Act(Norm(conv(h, edges)))) (residual identity); residual linear
    = Dropout(Act(Norm(conv(h) + W_r h))); none = no skip. ffn adds h <- Norm(h + W2(Dropout(ReLU(W1 h)))).'''

    def __init__(self, operator, hidden, mp_edge_dim, r, deg_hist=None, local='gin'):
        super().__init__()
        self.use_edges = mp_edge_dim > 0
        self.conv_in = make_conv(operator, hidden, mp_edge_dim, deg_hist, r['self_loops'], local, r['conv'])
        self.act = make_activation(r['activation'])
        self.dropout = nn.Dropout(r['dropout'])
        self.norm = make_norm(r['norm'], hidden)
        self.ffn = None
        if r['ffn']:
            self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Dropout(r['dropout']),
                                     nn.Linear(2 * hidden, hidden))
            self.ffn_norm = make_norm(r['norm'] if r['norm'] != 'none' else 'layer', hidden)
        self.residual = r['residual']
        if self.residual == 'linear':
            self.residual_proj = nn.Linear(hidden, hidden)

    def forward(self, h, edge_index, edge_attr):
        out = self.conv_in(h, edge_index, edge_attr=edge_attr) if self.use_edges else self.conv_in(h, edge_index)
        if self.residual == 'linear':
            out = out + self.residual_proj(h)
        out = self.dropout(self.act(self.norm(out)))
        h = h + out if self.residual == 'identity' else out
        if self.ffn is not None:
            h = self.ffn_norm(h + self.ffn(h))
        return h


class EdgeClassifier(nn.Module):
    '''node_proj Linear(node_dim -> hidden) + Act -> num_layers MPLayers -> readout on the seed edge:
    concat [h_src || h_dst || e_seed] -> (readout_layers - 1) x (Linear -> Act -> Dropout) -> Linear 1 (logit).'''

    def __init__(self, operator, node_dim, mp_edge_dim, readout_edge_dim, r, deg_hist=None, local='gin'):
        super().__init__()
        hidden = r['hidden']
        self.node_proj = nn.Linear(node_dim, hidden)
        self.act = make_activation(r['activation'])
        self.layers = nn.ModuleList([MPLayer(operator, hidden, mp_edge_dim, r, deg_hist, local)
                                     for _ in range(r['num_layers'])])
        width = max(1, int(round(r['readout_width'] * hidden)))
        in_dim, readout = 2 * hidden + readout_edge_dim, []
        for _ in range(r['readout_layers'] - 1):
            readout += [nn.Linear(in_dim, width), make_activation(r['activation']), nn.Dropout(r['dropout'])]
            in_dim = width
        self.classifier = nn.Sequential(*readout, nn.Linear(in_dim, 1))

    def forward(self, x, edge_index, edge_attr, seed_index, seed_edge_feats):
        h = self.act(self.node_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        z = torch.cat([h[seed_index[0]], h[seed_index[1]], seed_edge_feats], dim=1)
        return self.classifier(z).squeeze(-1)


def build_model(operator, node_dim, mp_edge_dim, readout_edge_dim, recipe, deg_hist=None, local='gin'):
    '''The template for one operator, feature configuration and recipe, on the device. local = the operator inside gps.'''
    return EdgeClassifier(operator, node_dim, mp_edge_dim, readout_edge_dim, check_recipe(recipe), deg_hist, local).to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def invariant_params(model):
    '''Parameters whose shape does not depend on the feature widths (identical across the configs of a family):
    everything but the conv's edge projection and the readout's first Linear; node_proj counted at the base width 6.'''
    width_dependent = r'\.lin\.|lin_edge|edge_encoder|pre_nns\.\d+\.0\.|classifier\.0\.'   # one pre_nns per PNA tower
    total = 0
    for name, p in model.named_parameters():
        if name == 'node_proj.weight':
            total += model.node_proj.out_features * BASE_NODE_DIM
        elif not re.search(width_dependent, name):
            total += p.numel()
    return total
