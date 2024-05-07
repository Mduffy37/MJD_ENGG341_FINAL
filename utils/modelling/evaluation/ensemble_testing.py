import os
import pickle
import pandas as pd
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def test_ensemble(X_test, y_test, ensemble_id):
    model_eval_results_path = 'data_store/modelling/model_performance/ensemble_evaluation_results.csv'
    model_eval_results = pd.read_csv(model_eval_results_path)

    print(f'Testing {ensemble_id}...')
    model_path = f'data_store/modelling/models/testing_models/{ensemble_id}.pkl'

    if not os.path.exists(model_path):
        print(f'{ensemble_id} not found. Skipping...')
        return

    model = pickle.load(open(model_path, 'rb'))

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100
    r2 = r2_score(y_test, y_pred)

    model_eval_results = model_eval_results.append({
        'model_id': ensemble_id,
        'model_name': model.__class__.__name__,
        'test_mae': mae,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_r2': r2,
        'openmc_time_taken': None,
        'openmc_mae': None,
        'openmc_mse': None,
        'openmc_rmse': None,
        'openmc_r2': None
    }, ignore_index=True)

    #model_eval_results.to_csv(model_eval_results_path, index=False)

    print(f'{ensemble_id} tested. MAE: {mae}, MSE: {mse}, RMSE: {rmse}, RMSPE: {rmspe}, R2: {r2}')


def test_ensemble_openmc(X_open_mc, y_open_mc, ensemble_id):
    model_eval_results_path = 'data_store/modelling/model_performance/ensemble_evaluation_results.csv'
    model_eval_results = pd.read_csv(model_eval_results_path)

    print(f'Testing {ensemble_id} with openmc data...')
    model_path = f'data_store/modelling/models/{ensemble_id}.pkl'

    if not os.path.exists(model_path):
        print(f'{ensemble_id} not found. Skipping...')
        return

    model = pickle.load(open(model_path, 'rb'))

    start_time = datetime.now()
    y_pred = model.predict(X_open_mc)
    time_taken = datetime.now() - start_time

    mae = mean_absolute_error(y_open_mc, y_pred)
    mse = mean_squared_error(y_open_mc, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_open_mc, y_pred)

    model_eval_results.loc[model_eval_results['model_id'] == ensemble_id, 'openmc_time_taken'] = time_taken.total_seconds()
    model_eval_results.loc[model_eval_results['model_id'] == ensemble_id, 'openmc_mae'] = mae
    model_eval_results.loc[model_eval_results['model_id'] == ensemble_id, 'openmc_mse'] = mse
    model_eval_results.loc[model_eval_results['model_id'] == ensemble_id, 'openmc_rmse'] = rmse
    model_eval_results.loc[model_eval_results['model_id'] == ensemble_id, 'openmc_r2'] = r2

    model_eval_results.to_csv(model_eval_results_path, index=False)

    print(f'{ensemble_id} tested. MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')