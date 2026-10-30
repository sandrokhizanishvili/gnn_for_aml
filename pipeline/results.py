'''What a run writes and where: run names and folders, results.json, predictions.csv, curves.png, the resolved
config, atomic JSON writes, the trial log of the tabular searches.'''
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import precision_recall_curve

from pipeline.data import SPLITS
from pipeline.metrics import METRICS

matplotlib.use('Agg')   # files only, never a window
MODE_DIR = {'train': 'train', 'tune': 'tuning', 'final': 'final'}


# ---------------------------------------------------------------- names and folders

def run_name(model, feature_config, node_encoding='none', gfp_sheet='default'):
    '''<model>_mp-<mp>_readout-<readout>_dir-in[_enc-<encoding>][_gfp-<sheet>] (the thesis naming).'''
    name = f"{model}_mp-{feature_config['mp']}_readout-{feature_config['readout']}_dir-in"
    if node_encoding != 'none':
        name += f'_enc-{node_encoding}'
    if gfp_sheet != 'default':
        name += f'_gfp-{gfp_sheet}'
    return name


def run_root(outputs, mode, model, name):
    '''outputs/<train | tuning | final>/<model>/<run>: shared by every seed and trial of one run.'''
    return f'{outputs}/{MODE_DIR[mode]}/{model}/{name}'


def run_dir(cfg, name):
    '''The folder of one job: run_root + /t<trial>/seed-<s> (a GNN tuning job), /seed-<s> (train, final), or
    run_root itself (a tabular search, one job).'''
    root = run_root(cfg['paths']['outputs'], cfg['mode'], cfg['model']['name'], name)
    if cfg['mode'] == 'tune':
        return root if cfg['model']['kind'] == 'tabular' else f"{root}/t{cfg['trial']:03d}/seed-{cfg['seed']}"
    return f"{root}/seed-{cfg['seed']}"


# ---------------------------------------------------------------- files

def write_json_atomic(path, obj):
    '''json.dump through a temporary file + os.replace: a killed job never leaves a half-written file.'''
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_config(cfg, out_dir):
    '''config.yaml: the resolved configuration of the run.'''
    OmegaConf.save(OmegaConf.create(cfg), f'{out_dir}/config.yaml')


def result_row(name, cfg, best_epoch, threshold, n_params, n_invariant, metrics):
    '''The flat row of results.json and the batch summaries: the knobs, the run fields, <split>_<metric>.'''
    row = {'run': name, 'operator': cfg['model']['name'], 'mp': cfg['feature_config']['mp'],
           'readout': cfg['feature_config']['readout'], 'mp_direction': 'in', 'temporal_sampling': False,
           'node_enc': cfg['node_encoding'], 'gfp_variant': cfg['gfp_sheet'], 'seed': cfg['seed'],
           'best_epoch': best_epoch, 'threshold': threshold, 'params': n_params, 'invariant_params': n_invariant}
    row.update({f'{split}_{m}': v for split in SPLITS for m, v in metrics[split].items()})
    return row


def save_run(out_dir, row, hyperparameters, extra=None):
    '''results.json = the flat row + the hyperparameters + extras (written last: it marks a finished run).'''
    write_json_atomic(f'{out_dir}/results.json', dict(row, recipe=hyperparameters, **(extra or {})))
    print(pd.DataFrame({split: {m: row[f'{split}_{m}'] for m in METRICS} for split in SPLITS}).round(4))


def save_predictions(out_dir, graphs, preds, threshold):
    '''predictions.csv: one row per evaluated edge of every split; pred = prob >= the validation-chosen threshold.'''
    frames = []
    for split in SPLITS:
        graph, edge_id = graphs[split], torch.as_tensor(preds[split]['edge_id'])
        prob = preds[split]['prob'].astype(np.float32)
        frames.append(pd.DataFrame({'split': split, 'edge_id': edge_id.numpy(),
                                    'src': graph.edge_index[0, edge_id].numpy(), 'dst': graph.edge_index[1, edge_id].numpy(),
                                    'edge_time': graph.edge_time[edge_id].numpy(), 'y': preds[split]['y'], 'prob': prob,
                                    'pred': (prob >= threshold).astype(int)}))
    table = pd.concat(frames).sort_values('edge_id')
    assert table['edge_id'].is_unique, 'edge_id must identify every scored edge once'
    table.to_csv(f'{out_dir}/predictions.csv', index=False)
    print(f'predictions: {len(table):,} rows -> {out_dir}/predictions.csv')


def plot_curves(history, title, path, best_epoch=None, test=None):
    '''curves.png: one panel per epoch series (loss, F1, PR-AUC, learning rate, threshold), a dashed line at
    best_epoch, and the test precision-recall curve (test = (y, prob, pr_auc)).'''
    df = pd.DataFrame(history)
    panels = [k for k in ('loss', 'f1', 'pr_auc', 'lr', 'threshold') if f'val_{k}' in df or k in df]
    n = len(panels) + (test is not None)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6.5 * ((n + 1) // 2), 8))
    axes = axes.flatten()
    fig.suptitle(title, fontweight='bold')
    for axis, key in zip(axes, panels):
        series = [f'train_{key}', f'val_{key}', key] + (['train_loss_batches'] if key == 'loss' else [])
        for col, marker in zip(series, 'osdx'):
            if col in df:
                axis.plot(df['epoch'], df[col], marker=marker, label=col)
        if best_epoch:
            axis.axvline(best_epoch, color='gray', linestyle='--', label=f'best epoch = {best_epoch}')
        axis.set_title(key)
        axis.set_xlabel('epoch')
        axis.legend()
    if test is not None:
        y, p, pr_auc = test
        precision, recall, _ = precision_recall_curve(y, p)
        axes[len(panels)].plot(recall, precision, label=f'test AP = {pr_auc:.3f}')
        axes[len(panels)].axhline(y.mean(), color='gray', linestyle='--', label='random')
        axes[len(panels)].set_title('test precision-recall')
        axes[len(panels)].set_xlabel('recall')
        axes[len(panels)].legend()
    for axis in axes[n:]:
        axis.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- the trial log of a tabular search

def check_draw(path, configs):
    '''configs.json: written by the first run; a rerun must reproduce the same seeded draw.'''
    if os.path.exists(path):
        assert {int(k): v for k, v in json.load(open(path)).items()} == configs, 'configs.json differs from the seeded draw'
    else:
        json.dump(configs, open(path, 'w'), indent=1)


def log_trial(path, trials, row):
    '''Append one finished trial to trials.csv (so a stopped search resumes) and to the in-memory table.'''
    new = pd.DataFrame([row])
    new.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return pd.concat([trials, new], ignore_index=True) if len(trials) else new


def read_trials(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
