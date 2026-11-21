'''The search every model shares (mode tune): an Optuna study with the TPE sampler and the Hyperband pruner, kept in
the journal file study.log, so a stopped search continues and several workers can share one study.'''
import os

import matplotlib
import matplotlib.pyplot as plt
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

matplotlib.use('Agg')


def open_study(study_dir, search, cap, worker=0):
    '''The study saved in study_dir, created on first use. cap = the epochs or rounds of a full trial.'''
    path = f'{study_dir}/study.log'   # the open lock: the default one needs symlinks, which Windows refuses
    storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    pruner = optuna.pruners.HyperbandPruner(min_resource=search['first_check'], max_resource=cap,
                                            reduction_factor=search['reduction_factor'])
    study = optuna.create_study(study_name=os.path.basename(study_dir), storage=storage, pruner=pruner,
                                direction='maximize', load_if_exists=True)
    # the seed moves with the worker and with the trials already there: a restarted worker does not repeat its draws
    seed = search['sampler_seed'] + 1000 * worker + len(study.trials)
    study.sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=search['startup_trials'])
    return study


def suggest(trial, space):
    '''The trial's value of every parameter: [int | float | log, low, high] is a range, any other list is choices.'''
    values = {}
    for name, spec in space.items():
        if spec[0] == 'int':
            values[name] = trial.suggest_int(name, spec[1], spec[2])
        elif spec[0] in ('float', 'log'):
            values[name] = trial.suggest_float(name, spec[1], spec[2], log=spec[0] == 'log')
        else:
            values[name] = trial.suggest_categorical(name, spec)
    return values


def run_search(study, study_dir, space, start, n_trials, worker, run_trial):
    '''Runs trials until the study has n_trials; worker 0 queues the start trial, if there is one.
    run_trial(values, trial_dir, report) -> (score, info); score = None when the trial cannot win.
    report(best score so far, step) stops the trial when the pruner says so.'''
    stopped = [t for t in study.get_trials(states=(TrialState.RUNNING,)) if t.user_attrs.get('worker') == worker]
    if worker == 0 and start and not any(t.user_attrs.get('start') for t in study.trials):
        study.enqueue_trial(start, user_attrs={'start': True})

    while stopped or len(study.trials) < n_trials:
        if stopped:   # this worker was stopped during the trial: it continues under its number, which sets its bracket
            before = stopped.pop()
            trial, reported = optuna.trial.Trial(study, before._trial_id), max(before.intermediate_values, default=0)
        else:
            trial, reported = study.ask(), 0
            trial.set_user_attr('worker', worker)
        values = suggest(trial, space)   # a continued trial gets its stored values back
        folder = f't{trial.number:03d}'

        def report(score, step):
            if step > reported:   # the steps up to `reported` were checked before the stop
                trial.report(score, step)
                if trial.should_prune() and not trial.user_attrs.get('start'):
                    raise optuna.TrialPruned()

        print(f'--- trial {trial.number}: {values}', flush=True)
        try:
            score, info = run_trial(values, f'{study_dir}/{folder}', report)
        except optuna.TrialPruned:
            score, info = None, {}
        for key, value in info.items():
            trial.set_user_attr(key, value)
        state = TrialState.COMPLETE if score is not None else TrialState.PRUNED
        study.tell(trial, score, state=state)
        print(f'trial {trial.number} {state.name}' + (f': {score:.4f}' if score is not None else ''), flush=True)
        if os.path.exists(f'{study_dir}/{folder}/resume.pt'):   # the checkpoint of a GNN trial, no longer needed
            os.remove(f'{study_dir}/{folder}/resume.pt')


def save_summary(study, study_dir):
    '''trials.csv (one row per trial) and tuning_curves.png (every trial's best validation PR-AUC so far).'''
    study.trials_dataframe().to_csv(f'{study_dir}/trials.csv', index=False)
    complete = study.get_trials(states=(TrialState.COMPLETE,))
    winner = study.best_trial.number if complete else None
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in study.trials:
        steps = sorted(t.intermediate_values)
        is_winner = t.number == winner
        ax.plot(steps, [t.intermediate_values[s] for s in steps], color='C3' if is_winner else 'gray',
                linewidth=2.2 if is_winner else 0.8, alpha=1.0 if is_winner else 0.4,
                label=f'winner: trial {t.number} ({t.value:.3f})' if is_winner else None)
    ax.set_xlabel('epoch or round')
    ax.set_ylabel('best validation PR-AUC so far')
    ax.set_title(f'{study.study_name}: {len(complete)} full trials of {len(study.trials)}')
    if winner is not None:
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{study_dir}/tuning_curves.png', dpi=120)
    plt.close(fig)
