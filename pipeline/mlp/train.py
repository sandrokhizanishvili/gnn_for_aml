'''MLP training at one configuration: Adam at a constant learning rate for a given number of epochs, train and val
scored every epoch.'''
import time

import torch
import torch.nn as nn

from pipeline.data import set_seed
from pipeline.gnn.model import count_params, device
from pipeline.lightgbm.model import LOSS_WEIGHT
from pipeline.metrics import epoch_row
from pipeline.mlp.model import MLP, predict_mlp


def fit_mlp(knobs, X, y, rows, fit_rows, epochs, seed, report=None):
    '''X: the standardised matrix on the device; y: the 0/1 labels (numpy); fit_rows: the slice trained on (the train
    rows in a search, train + val in the final refit). A search passes report(best score so far, epoch), which may
    stop the trial. Returns (model, history).'''
    set_seed(seed)
    model = MLP(X.shape[1], knobs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=knobs['lr'], weight_decay=knobs['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(knobs['pos_weight']), device=device))
    tr, va = rows['train'], rows['val']
    y_train, y_val = y[tr].astype(int), y[va].astype(int)
    y = torch.from_numpy(y).float().to(device)
    print(f"{knobs['shape']} hidden {knobs['hidden']} depth {knobs['depth']} | params {count_params(model):,} | seed {seed}")
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order, losses = fit_rows.start + torch.randperm(fit_rows.stop - fit_rows.start, device=device), []
        for i in range(0, len(order), knobs['batch_size']):
            idx = order[i:i + knobs['batch_size']]
            optimizer.zero_grad()
            loss = criterion(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        row = epoch_row(epoch, y_train, predict_mlp(model, X, tr), y_val, predict_mlp(model, X, va), LOSS_WEIGHT, t0)
        row.update(train_loss_batches=float(torch.stack(losses).mean()), lr=knobs['lr'])
        history.append(row)
        best = max(best, row['val_pr_auc'])
        print(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} / {row['val_loss']:.4f} | F1 {row['train_f1']:.4f} / {row['val_f1']:.4f} "
              f"@ {row['threshold']:.3f} | PR-AUC {row['train_pr_auc']:.4f} / {row['val_pr_auc']:.4f} | {row['seconds']:.1f}s", flush=True)
        if report is not None and epoch < epochs:   # after the last epoch the trial is complete
            report(best, epoch)
    return model, history
