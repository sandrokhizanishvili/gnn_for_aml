'''MLP training at one configuration: Adam at a constant learning rate, train and val scored every epoch, the
weights of the best validation PR-AUC epoch kept, stop after hp['patience'] epochs without a gain.'''
import time

import torch
import torch.nn as nn

from pipeline.data import set_seed
from pipeline.gnn.model import count_params, device
from pipeline.lightgbm.model import LOSS_WEIGHT
from pipeline.metrics import epoch_row
from pipeline.mlp.model import MLP, predict_mlp


def fit_mlp(knobs, hp, X, y, rows, seed):
    '''Returns (model, history, best epoch).'''
    set_seed(seed)
    model = MLP(X.shape[1], knobs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=knobs['lr'], weight_decay=knobs['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(knobs['pos_weight']), device=device))
    tr, va = rows['train'], rows['val']
    y_train, y_val = y[tr].cpu().numpy().astype(int), y[va].cpu().numpy().astype(int)
    print(f"{knobs['shape']} hidden {knobs['hidden']} depth {knobs['depth']} | params {count_params(model):,} | seed {seed}")
    history, best_epoch, best_score, best_state = [], 0, -1.0, None
    for epoch in range(1, hp['epochs'] + 1):
        t0 = time.time()
        model.train()
        order, losses = tr.start + torch.randperm(tr.stop - tr.start, device=device), []
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
        if row['val_pr_auc'] > best_score:
            best_epoch, best_score = epoch, row['val_pr_auc']
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} / {row['val_loss']:.4f} | F1 {row['train_f1']:.4f} / {row['val_f1']:.4f} "
              f"@ {row['threshold']:.3f} | PR-AUC {row['train_pr_auc']:.4f} / {row['val_pr_auc']:.4f} | {row['seconds']:.1f}s"
              + ('   * new best' if epoch == best_epoch else ''))
        if epoch - best_epoch >= hp['patience']:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch
