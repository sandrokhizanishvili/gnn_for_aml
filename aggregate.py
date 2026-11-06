'''The login-node step of the GNN search (no GPU, minutes): scores every trial whose seeds are done, runs the checks,
freezes the winner of every (model, feature_config) pair into recipe.json, writes the summaries, prints the progress.

    python aggregate.py machine=leonardo        (same config as run.py with hyperparameters=gnn/space)
'''
import glob
import json
import os
import sys

import pandas as pd
from hydra import compose, initialize
from omegaconf import OmegaConf

from pipeline.gnn import tune
from pipeline.gnn.model import RECIPE_KEYS
from pipeline.results import run_name, run_root, save_seed_summary, write_json_atomic

CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf')


def tuning_pair(cfg, model, feature_config, trials_drawn, base):
    '''Checks, summaries and (once every trial is in) the frozen winner of one (model, feature_config).'''
    s = cfg['hyperparameters']['search']
    rdir = run_root(cfg['paths']['outputs'], 'tune', model, run_name(model, feature_config))
    for trial in trials_drawn:
        tune.finalise_trial(f"{rdir}/t{trial['id']:03d}", s['tuning_seeds'])
    trials = tune.load_trials(rdir)
    for trial in trials:
        expected = tune.trial_recipe(base, trials_drawn[trial['trial_id']])
        tune.check_trial(trial, expected, f"{rdir}/t{trial['trial_id']:03d}", s['tuning_seeds'], base['epochs'])
    seeds_done = len(glob.glob(f'{rdir}/t*/seed-*/done.json'))
    status = pd.Series([x['status'] for x in trials], dtype=object).value_counts().to_dict()
    print(f"{model} {feature_config['mp']}/{feature_config['readout']}: seeds done {seeds_done}/{len(s['tuning_seeds']) * s['n_trials']}, "
          f"trials scored {len(trials)}/{s['n_trials']} {status}, checks passed")
    if not trials:
        return
    winner = tune.choose_best_trial(trials, base['select_metric']) if len(trials) == s['n_trials'] else None
    recipe_path = f'{rdir}/recipe.json'
    if winner is not None and os.path.exists(recipe_path):
        assert json.load(open(recipe_path))['trial_id'] == winner['trial_id'], f'{recipe_path} is not the best trial'
    elif winner is not None:
        tune.save_recipe(recipe_path, winner, s['final_seeds'])
    winner_id = winner['trial_id'] if winner else None
    tune.trials_table(trials, s['space'], winner_id).to_csv(f'{rdir}/trials_summary.csv', index=False)
    tune.plot_val_curves(trials, winner_id, f'{rdir}/tuning_curves.png', base['select_metric'])


def final_pair(cfg, model, feature_config):
    '''Checks and the mean +- std summary of the final-retrain seeds of one pair finished so far.'''
    s, outputs = cfg['hyperparameters']['search'], cfg['paths']['outputs']
    recipe_path = f"{run_root(outputs, 'tune', model, run_name(model, feature_config))}/recipe.json"
    if not os.path.exists(recipe_path):
        return
    frozen = tune.load_recipe(recipe_path)
    pair_dir = run_root(outputs, 'final', model, run_name(model, feature_config))
    rows = []
    for seed in s['final_seeds']:
        if os.path.exists(f'{pair_dir}/seed-{seed}/results.json'):
            results = json.load(open(f'{pair_dir}/seed-{seed}/results.json'))
            tune.check_final(results, pd.read_csv(f'{pair_dir}/seed-{seed}/history.csv'), frozen, seed)
            rows.append({k: v for k, v in results.items() if k not in ('recipe', 'threshold_sensitivity')})
    print(f"  final retrain: {len(rows)}/{len(s['final_seeds'])} seeds done, checks passed")
    if rows:
        save_seed_summary(pair_dir, rows)


def main():
    with initialize(version_base=None, config_path='conf'):   # the jobs' config, with the search file selected
        cfg = OmegaConf.to_container(compose(config_name='config', overrides=['hyperparameters=gnn/space'] + sys.argv[1:]), resolve=True)
    hp = cfg['hyperparameters']
    s, base = hp['search'], {k: hp[k] for k in RECIPE_KEYS}
    trials_drawn = tune.draw_trials(s)
    os.makedirs(f"{cfg['paths']['outputs']}/tuning", exist_ok=True)
    write_json_atomic(f"{cfg['paths']['outputs']}/tuning/trials_space.json",   # the record of the pre-registered draw
                      dict(s, base_recipe=base, trials=trials_drawn))
    for model in s['models']:
        for name in s['feature_configs']:
            feature_config = OmegaConf.to_container(OmegaConf.load(f'{CONF}/feature_config/{name}.yaml'))
            tuning_pair(cfg, model, feature_config, trials_drawn, base)
            final_pair(cfg, model, feature_config)
    print('\nall checks passed')


if __name__ == '__main__':
    main()
