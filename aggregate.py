'''The login-node step of the GNN search (no GPU, minutes): for every (model, feature_config) it prints the progress of
the study, runs the checks, writes trials.csv and tuning_curves.png, and summarises the final-retrain seeds done so far.

    python aggregate.py machine=leonardo        (same config as run.py with hyperparameters=gnn/space)
'''
import collections
import json
import os
import sys

import pandas as pd
from hydra import compose, initialize
from omegaconf import OmegaConf
from optuna.trial import TrialState

from pipeline import tune
from pipeline.gnn import tune as gnn_tune
from pipeline.results import run_name, run_root, save_seed_summary

CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf')


def search_pair(cfg, model, feature_config):
    '''Progress, checks and summaries of the study of one (model, feature_config).'''
    hp = cfg['hyperparameters']
    rdir = run_root(cfg['paths']['outputs'], 'tune', model, run_name(model, feature_config))
    if not os.path.exists(f'{rdir}/study.log'):
        print(f"{model} {feature_config['mp']}/{feature_config['readout']}: search not started")
        return
    study = tune.open_study(rdir, hp['search'], hp['epochs'])
    states = collections.Counter(t.state.name for t in study.trials)
    for trial in study.get_trials(states=(TrialState.COMPLETE,)):
        gnn_tune.check_trial(trial, f"{rdir}/t{trial.number:03d}", hp['epochs'])
    print(f"{model} {feature_config['mp']}/{feature_config['readout']}: {dict(states)} of {hp['search']['n_trials']} trials, checks passed")
    tune.save_summary(study, rdir)
    if os.path.exists(f'{rdir}/recipe.json'):
        frozen = json.load(open(f'{rdir}/recipe.json'))
        assert frozen['trial'] == study.best_trial.number, f'{rdir}/recipe.json is not the best trial'
        assert frozen['test_split_loaded'] is False, f'{rdir}: the search loaded the test split'
        print(f"  winner: trial {frozen['trial']}, val PR-AUC {frozen['val_pr_auc']:.4f} at epoch {frozen['best_epoch']}")


def final_pair(cfg, model, feature_config):
    '''Checks and the mean +- std summary of the final-retrain seeds of one pair finished so far.'''
    final_seeds, outputs = cfg['hyperparameters']['search']['final_seeds'], cfg['paths']['outputs']
    recipe_path = f"{run_root(outputs, 'tune', model, run_name(model, feature_config))}/recipe.json"
    if not os.path.exists(recipe_path):
        return
    frozen = json.load(open(recipe_path))
    pair_dir = run_root(outputs, 'final', model, run_name(model, feature_config))
    rows = []
    for seed in final_seeds:
        if os.path.exists(f'{pair_dir}/seed-{seed}/results.json'):
            results = json.load(open(f'{pair_dir}/seed-{seed}/results.json'))
            gnn_tune.check_final(results, pd.read_csv(f'{pair_dir}/seed-{seed}/history.csv'), frozen, seed)
            rows.append({k: v for k, v in results.items() if k not in ('recipe', 'threshold_sensitivity')})
    print(f'  final retrain: {len(rows)}/{len(final_seeds)} seeds done, checks passed')
    if rows:
        save_seed_summary(pair_dir, rows)


def main():
    with initialize(version_base=None, config_path='conf'):   # the jobs' config, with the search file selected
        cfg = OmegaConf.to_container(compose(config_name='config', overrides=['hyperparameters=gnn/space'] + sys.argv[1:]), resolve=True)
    search = cfg['hyperparameters']['search']
    for model in search['models']:
        for name in search['feature_configs']:
            feature_config = OmegaConf.to_container(OmegaConf.load(f'{CONF}/feature_config/{name}.yaml'))
            search_pair(cfg, model, feature_config)
            final_pair(cfg, model, feature_config)
    print('\nall checks passed')


if __name__ == '__main__':
    main()
