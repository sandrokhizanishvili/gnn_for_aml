'''The MLP on the 93 inputs (no message passing): the network, its predictions, the input standardisation.'''
import numpy as np
import torch
import torch.nn as nn

from pipeline.gnn.model import make_norm   # the same norm layers as the GNN template

N_ONEHOT = 12              # the entity one-hots are the last 12 columns of the matrix


def standardise(X, rows, std_floor):
    '''Edge columns: (value - train mean) / train std, std floored; the one-hot columns stay 0 / 1. In place.'''
    n_edge = X.shape[1] - N_ONEHOT
    assert np.isin(X[:, n_edge:], (0.0, 1.0)).all(), 'the last 12 columns must be the entity one-hots'
    train_rows = X[rows['train'], :n_edge]
    mean = train_rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(train_rows.std(axis=0, dtype=np.float64), std_floor).astype(np.float32)
    X[:, :n_edge] -= mean
    X[:, :n_edge] /= std
    return X


class MLP(nn.Module):
    '''flat / funnel: [Linear -> Norm -> ReLU -> Dropout] per width, then Linear -> 1. resnet: Linear in -> hidden,
    `depth` residual blocks z + W2(Dropout(ReLU(Norm(W1 z)))), then Linear -> 1 (Gorishniy et al. 2021).'''

    def __init__(self, in_dim, k):
        super().__init__()
        hidden, depth = k['hidden'], k['depth']
        widths = [max(16, hidden // 2 ** i) for i in range(depth)] if k['shape'] == 'funnel' else [hidden] * depth
        layers, last = [], in_dim
        if k['shape'] == 'resnet':
            layers.append(nn.Linear(in_dim, hidden))
            self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(hidden, hidden), make_norm(k['norm'], hidden), nn.ReLU(),
                                                      nn.Dropout(k['dropout']), nn.Linear(hidden, hidden)) for _ in range(depth))
            last = hidden
        else:
            self.blocks = None
            for w in widths:
                layers += [nn.Linear(last, w), make_norm(k['norm'], w), nn.ReLU(), nn.Dropout(k['dropout'])]
                last = w
        self.stem, self.head = nn.Sequential(*layers), nn.Linear(last, 1)

    def forward(self, x):
        z = self.stem(x)
        for block in self.blocks or ():
            z = z + block(z)
        return self.head(z).squeeze(-1)


@torch.no_grad()
def predict_mlp(model, X, rows, chunk=65_536):
    '''Probabilities of the rows [rows.start, rows.stop), in chunks that fit the device.'''
    model.eval()
    probs = [torch.sigmoid(model(X[i:min(i + chunk, rows.stop)])).cpu() for i in range(rows.start, rows.stop, chunk)]
    return torch.cat(probs).numpy()
