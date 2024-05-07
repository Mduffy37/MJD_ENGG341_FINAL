import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

import utils.modelling.hyper_parameters as hp
import utils.modelling.cross_validation as cv


def complete_hyperparameter_tuning(models, X, y,
                                   overwrite_models=None,
                                   skip_models=None):
    if overwrite_models is None:
        overwrite_models = []
    if skip_models is None:
        skip_models = []

    file_path = 'data_store/modelling/hyper_parameter_tuning/hyper_parameter_tuning_results.csv'
    model_results = pd.read_csv(file_path)

    for model_id, model in models.items():
        if model_id in skip_models:
            continue

        model_results = _add_new_row(model_results, model_id, model)
        model_row = model_results.loc[model_results['model_id'] == model_id]

        if model_row['random_search_params'].isna().values[0] or model_id in overwrite_models:
            _complete_random_search(model_results, model_id, model, X, y)
            model_results.to_csv(file_path, index=False)

        if model_row['grid_search_params'].isna().values[0] or model_id in overwrite_models:
            _complete_grid_search(model_results, model_id, model, X, y)
            model_results.to_csv(file_path, index=False)

        if model_row['optuna_best_params'].isna().values[0] or model_id in overwrite_models:
            overwrite = _return_overwrite_model(model_id, overwrite_models)
            _complete_optuna_optimise(model_results, model_id, model, X, y, overwrite=overwrite)
            model_results.to_csv(file_path, index=False)


def _return_overwrite_model(model_id, overwrite_models):
    if model_id in overwrite_models:
        overwrite = True
        warnings.warn(f'Overwriting {model_id}')
    else:
        overwrite = False
    return overwrite


def _add_new_row(model_results, model_id, model):
    if model_id not in model_results['model_id'].values:
        new_row = pd.DataFrame({
            'model_id': [model_id],
            'model_name': [model.__name__]
        })
        return pd.concat([model_results, new_row], ignore_index=True)
    else:
        return model_results


def _complete_random_search(model_results, model_id, model, X, y):
    start = datetime.now()
    random_search_params = hp.random_search_optimise(model_id, X, y)
    model_results.loc[model_results['model_id'] == model_id, 'random_search_params'] = json.dumps(random_search_params)

    random_search_model = model(**random_search_params)
    random_search_cv_scores = cv.cross_validation(X, y, random_search_model)
    model_results.loc[model_results['model_id'] == model_id, 'random_search_time_taken'] = datetime.now() - start
    model_results.loc[model_results['model_id'] == model_id, 'random_search_cv_mae'] = random_search_cv_scores['mae']
    model_results.loc[model_results['model_id'] == model_id, 'random_search_cv_mse'] = random_search_cv_scores['mse']
    model_results.loc[model_results['model_id'] == model_id, 'random_search_cv_rmse'] = random_search_cv_scores['rmse']
    model_results.loc[model_results['model_id'] == model_id, 'random_search_cv_r2'] = random_search_cv_scores['r2']


def _complete_grid_search(model_results, model_id, model, X, y):
    start = datetime.now()
    grid_search_params = hp.grid_search_optimise(model_id, X, y)
    model_results.loc[model_results['model_id'] == model_id, 'grid_search_params'] = json.dumps(grid_search_params)

    grid_search_model = model(**grid_search_params)
    grid_search_cv_scores = cv.cross_validation(X, y, grid_search_model)
    model_results.loc[model_results['model_id'] == model_id, 'grid_search_time_taken'] = datetime.now() - start
    model_results.loc[model_results['model_id'] == model_id, 'grid_search_cv_mae'] = grid_search_cv_scores['mae']
    model_results.loc[model_results['model_id'] == model_id, 'grid_search_cv_mse'] = grid_search_cv_scores['mse']
    model_results.loc[model_results['model_id'] == model_id, 'grid_search_cv_rmse'] = grid_search_cv_scores['rmse']
    model_results.loc[model_results['model_id'] == model_id, 'grid_search_cv_r2'] = grid_search_cv_scores['r2']


def _complete_optuna_optimise(model_results, model_id, model, X, y, overwrite=False):
    start = datetime.now()
    optuna_results = hp.optuna_optimise(model_id, X, y,
                                        n_trials=1000,
                                        suppress_warnings=False,
                                        scoring='rmse',
                                        study_name=model_id,
                                        overwrite=overwrite)

    optuna_best_params = optuna_results['best_params']
    model_results.loc[model_results['model_id'] == model_id, 'optuna_best_params'] = json.dumps(optuna_best_params)

    optuna_model = model(**optuna_best_params)
    optuna_cv_scores = cv.cross_validation(X, y, optuna_model)
    model_results.loc[model_results['model_id'] == model_id, 'optuna_time_taken'] = datetime.now() - start
    model_results.loc[model_results['model_id'] == model_id, 'optuna_cv_mae'] = optuna_cv_scores['mae']
    model_results.loc[model_results['model_id'] == model_id, 'optuna_cv_mse'] = optuna_cv_scores['mse']
    model_results.loc[model_results['model_id'] == model_id, 'optuna_cv_rmse'] = optuna_cv_scores['rmse']
    model_results.loc[model_results['model_id'] == model_id, 'optuna_cv_r2'] = optuna_cv_scores['r2']
