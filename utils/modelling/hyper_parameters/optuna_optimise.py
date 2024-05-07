from functools import partial

import optuna

from utils.modelling.hyper_parameters.optuna_objectives import objective


def custom_callback(study, trial,
                    verbose=False,
                    early_stopping=True,
                    early_stopping_iter=250):
    if verbose:
        print(f"Trial {trial.number} finished with value {trial.value} - "
              f"the best is {study.best_value} from trial {study.best_trial.number}")

    if early_stopping:
        if len(study.trials) >= early_stopping_iter:
            if trial.number > study.best_trial.number + early_stopping_iter:
                print(f"Stopping after {early_stopping_iter} iterations with no improvement")
                study.stop()


def optuna_optimise(algorithm, X, y,
                    n_trials=5000,
                    study_name=None,
                    suppress_warnings=True,
                    verbose=False,
                    scoring=None,
                    early_stopping=True,
                    early_stopping_iter=100,
                    overwrite=False):

    if suppress_warnings:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if scoring is None:
        raise ValueError('scoring was not provided')
    elif scoring in ['mae', 'mse', 'rmse']:
        direction = 'minimize'
    elif scoring in ['r2']:
        direction = 'maximize'
    else:
        raise ValueError(f"scoring {scoring} not recognised, must be one of ['mae', 'mse', 'rmse', 'r2']")

    storage_path = 'data_store/modelling/hyper_parameter_tuning/optuna_store.db'

    if overwrite:
        try:
            optuna.delete_study(study_name=study_name, storage=f'sqlite:///{storage_path}')
        except:
            pass

    study = optuna.create_study(direction=direction,
                                storage=f'sqlite:///{storage_path}',
                                load_if_exists=True,
                                study_name=study_name)

    callback = [partial(custom_callback,
                        verbose=verbose,
                        early_stopping=early_stopping,
                        early_stopping_iter=early_stopping_iter)]

    existing_trials = len(study.trials)
    if existing_trials > n_trials:
        print(f"Study already has {len(study.trials)} trials. Skipping new trials")
    else:
        study.optimize(partial(objective, algorithm=algorithm, X=X, y=y, scoring=scoring),
                       n_trials=n_trials - existing_trials,
                       callbacks=callback,
                       timeout=5400)

    trial = study.best_trial

    if algorithm == 'mlp':
        hidden_layer_sizes = []
        layer_keys = sorted([key for key in trial.params.keys() if 'layer' in key])
        for key in layer_keys:
            hidden_layer_sizes.append(trial.params.pop(key))
        trial.params['hidden_layer_sizes'] = hidden_layer_sizes
        trial.params.pop('depth')

    results = {
        'best_score': trial.value,
        'best_params': trial.params
    }
    return results
