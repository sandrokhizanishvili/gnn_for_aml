'''GNN training: neighbour-sampling loaders, prediction, the epoch loop (validation threshold sweep, best
checkpoint, resume.pt after every epoch), the fixed-epoch retrain of the final mode, scoring of the three splits.'''
import os
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from pipeline.data import SPLITS, set_seed
from pipeline.gnn.model import device
from pipeline.metrics import compute_metrics, epoch_row, weighted_bce


# ---------------------------------------------------------------- loaders and prediction

def make_loader(graph, shuffle, mp_idx, readout_idx, r):
    '''Neighbour-sampling loader over a split's evaluated edges + their readout features (looked up by input_id).'''
    mp_graph = Data(x=graph.x, edge_index=graph.edge_index, edge_time=graph.edge_time, num_nodes=graph.num_nodes,
                    edge_attr=graph.edge_attr[:, mp_idx] if len(mp_idx) > 0 else None)
    seed_mask = graph.eval_mask
    seed_feats = graph.edge_attr[seed_mask][:, readout_idx]
    kwargs = dict(num_neighbors=r['num_neighbors'], batch_size=r['batch_size'], shuffle=shuffle,
                  edge_label_index=graph.edge_index[:, seed_mask], edge_label=graph.y[seed_mask].float())
    if r['num_workers'] > 0:
        kwargs.update(num_workers=int(r['num_workers']), persistent_workers=False)
    return LinkNeighborLoader(mp_graph, **kwargs), seed_feats


def make_loaders(graphs, mp_idx, readout_idx, r, splits=SPLITS):
    '''One (loader, seed_feats) pair per split; train is shuffled, val / test are not.'''
    return {split: make_loader(graphs[split], split == 'train', mp_idx, readout_idx, r) for split in splits}


def forward_batch(model, batch, seed_feats):
    batch = batch.to(device)
    feats = seed_feats[batch.input_id.cpu()].to(device)
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, feats), batch.edge_label


@torch.no_grad()
def predict(model, loader, seed_feats):
    '''Labels, probabilities and seed positions (input_id) of every seed edge of the loader.'''
    model.eval()
    labels, probs, positions = [], [], []
    iter(loader)   # discarded: the thesis runs drew one extra shuffle permutation here (tqdm), kept for the same random stream
    for batch in loader:
        logits, y = forward_batch(model, batch, seed_feats)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
        positions.append(batch.input_id.cpu().numpy())
    return np.concatenate(labels).astype(int), np.concatenate(probs), np.concatenate(positions)


def score_all_splits(model, loaders, graphs, threshold, seed):
    '''Score train, val and test once at the threshold -> {split: metrics}, {split: {edge_id, y, prob}}.'''
    set_seed(seed)   # the same neighbour samples every time a model is scored
    metrics, preds = {}, {}
    for split in SPLITS:
        loader, seed_feats = loaders[split]
        y, p, positions = predict(model, loader, seed_feats)
        eval_positions = graphs[split].eval_mask.nonzero().squeeze(1).numpy()
        metrics[split] = compute_metrics(y, p, threshold)
        preds[split] = {'edge_id': eval_positions[positions], 'y': y, 'prob': p}
    return metrics, preds


# ---------------------------------------------------------------- the epoch loops

def train_epoch(model, loader, feats, optimizer, criterion):
    '''One pass over the loader in train mode; returns the mean mini-batch loss.'''
    model.train()
    total, count = 0.0, 0
    iter(loader)   # discarded: same random stream as the thesis runs (see predict)
    for batch in loader:
        optimizer.zero_grad()
        logits, y = forward_batch(model, batch, feats)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total, count = total + float(loss.detach()) * len(y), count + len(y)
    return total / max(1, count)


def make_optimizer(model, r):
    '''Adam + the recipe's schedule: cosine over the epochs, or constant.'''
    optimizer = torch.optim.Adam(model.parameters(), lr=r['lr'], weight_decay=r['weight_decay'])
    if r['lr_schedule'] == 'constant':
        return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=r['epochs'])


def save_resume(path, model, optimizer, scheduler, loop_state):
    '''resume.pt: model, optimizer, scheduler, every RNG state and the loop state, written atomically.'''
    rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
           'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    state = dict(loop_state, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng)
    tmp = f'{path}.tmp{os.getpid()}'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(path, model, optimizer, scheduler):
    '''The inverse of save_resume; returns the loop state, or None when there is nothing to resume from.'''
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    print(f"resumed from {path}: {len(state['history'])} epochs done")
    return state


def train_one_run(model, loaders, out_dir, r, save_best=True):
    '''The epoch loop: train + val scored every epoch, threshold swept on val, best epoch by r['select_metric']
    (best.pt when save_best), resumable. Returns (history rows, best_epoch, best_threshold).'''
    metric = r['select_metric']
    train_loader, train_feats = loaders['train']
    val_loader, val_feats = loaders['val']
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
    optimizer, scheduler = make_optimizer(model, r)
    history, stopped, best = [], False, (-1.0, 0, 0.5)   # best = (score, epoch, threshold)
    resume_path = f'{out_dir}/resume.pt'
    state = load_resume(resume_path, model, optimizer, scheduler)
    if state is not None:
        history, stopped, best = state['history'], state['stopped'], tuple(state['best'])
    for epoch in range(len(history) + 1, r['epochs'] + 1):
        if stopped:
            break
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        loss_batches = train_epoch(model, train_loader, train_feats, optimizer, criterion)
        y_train, p_train, _ = predict(model, train_loader, train_feats)
        y_val, p_val, _ = predict(model, val_loader, val_feats)
        if not (np.isfinite(p_val).all() and np.isfinite(p_train).all()):
            print(f'non-finite predictions at epoch {epoch}: the run diverged, training stops here')
            stopped = True   # no early stopping otherwise
        else:
            row = epoch_row(epoch, y_train, p_train, y_val, p_val, r['pos_weight'], t0)
            row.update(train_loss_batches=loss_batches, lr=lr_now)
            threshold = row['threshold']
            is_best = row[f'val_{metric}'] > best[0]
            if is_best:
                best = (row[f'val_{metric}'], epoch, threshold)
                if save_best:
                    torch.save(model.state_dict(), f'{out_dir}/best.pt')
            for split, y, p in (('train', y_train, p_train), ('val', y_val, p_val)):   # every metric, per epoch
                for name, value in compute_metrics(y, p, threshold).items():
                    row.setdefault(f'{split}_{name}', value)
            history.append(row)
            print(f"epoch {epoch:03d} | loss {row['train_loss']:.4f} / {row['val_loss']:.4f} | "
                  f"F1 {row['train_f1']:.4f} / {row['val_f1']:.4f} @ {threshold:.3f} | "
                  f"PR-AUC {row['train_pr_auc']:.4f} / {row['val_pr_auc']:.4f} | {row['seconds']:.0f}s | "
                  f"RAM {psutil.Process().memory_info().rss / 2**30:.1f}GB   (train / val)"
                  + ('   * new best' if is_best else ''), flush=True)
            scheduler.step()
        save_resume(resume_path, model, optimizer, scheduler, dict(history=history, stopped=stopped, best=best))
    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    print(f'best validation {metric} {best[0]:.4f} at epoch {best[1]} (threshold {best[2]:.3f}) | {len(history)} of {r["epochs"]} epochs run')
    return history, best[1], best[2]


def train_fixed_epochs(model, fit_loader, out_dir, r, stop_epoch, eval_loaders):
    '''The final retrain: stop_epoch epochs on the train + val loader, no model selection; train and val scored for
    the curves only (val is training data here). Resumable. Writes final.pt, history.csv.'''
    loader, feats = fit_loader
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([r['pos_weight']], device=device))
    optimizer, scheduler = make_optimizer(model, r)
    resume_path = f'{out_dir}/resume.pt'
    state = load_resume(resume_path, model, optimizer, scheduler)
    history = state['history'] if state is not None else []
    for epoch in range(len(history) + 1, stop_epoch + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        loss = train_epoch(model, loader, feats, optimizer, criterion)
        row = {'epoch': epoch, 'train_loss_batches': loss, 'lr': lr_now}
        for split, (eval_loader, eval_feats) in eval_loaders.items():
            y, p, _ = predict(model, eval_loader, eval_feats)
            assert np.isfinite(p).all(), f'non-finite {split} predictions at epoch {epoch}: the retrain diverged'
            row[f'{split}_loss'] = weighted_bce(y, p, r['pos_weight'])
            row[f'{split}_pr_auc'] = float(average_precision_score(y, p))
        row['seconds'] = time.time() - t0
        history.append(row)
        print(f"epoch {epoch:03d}/{stop_epoch} | batch loss {loss:.4f} | lr {lr_now:.2e} | "
              f"train PR-AUC {row['train_pr_auc']:.4f} | val (seen) PR-AUC {row['val_pr_auc']:.4f} | {row['seconds']:.0f}s", flush=True)
        scheduler.step()
        save_resume(resume_path, model, optimizer, scheduler, dict(history=history))
    torch.save(model.state_dict(), f'{out_dir}/final.pt')
    pd.DataFrame(history).to_csv(f'{out_dir}/history.csv', index=False)
    return history
