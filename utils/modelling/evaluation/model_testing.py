import os
import pickle
import pandas as pd
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def test_models(X_test, y_test, default_algos: dict):
    model_eval_results_path = 'data_store/modelling/model_performance/model_evaluation_results.csv'
    model_eval_results = pd.read_csv(model_eval_results_path)

    for model_id, model in default_algos.items():
        print(f'Testing {model_id}...')
        model_path = f'data_store/modelling/models/{model_id}.pkl'

        if not os.path.exists(model_path):
            print(f'{model_id} not found. Skipping...')
            continue

        if model_eval_results.loc[model_eval_results['model_id'] == model_id, 'test_r2'].notnull().values[0]:
            print(f'{model_id} already tested. Skipping...')
            continue

        model = pickle.load(open(model_path, 'rb'))

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        model_eval_results.loc[model_eval_results['model_id'] == model_id, 'test_mae'] = mae
        model_eval_results.loc[model_eval_results['model_id'] == model_id, 'test_mse'] = mse
        model_eval_results.loc[model_eval_results['model_id'] == model_id, 'test_rmse'] = rmse
        model_eval_results.loc[model_eval_results['model_id'] == model_id, 'test_r2'] = r2

        model_eval_results.to_csv(model_eval_results_path, index=False)

        print(f'{model_id} tested. MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')


def test_models_openmc(X_open_mc, y_open_mc, default_algos: dict):
    model_eval_results_path = 'data_store/modelling/model_performance/model_evaluation_results.csv'
    model_eval_results = pd.read_csv(model_eval_results_path)

    for model_id, model in default_algos.items():
        print(f'Testing {model_id} with openmc data...')
        model_path = f'data_store/modelling/models/{model_id}.pkl'

        if not os.path.exists(model_path):
            print(f'{model_id} not found. Skipping...')
            continue

        model = pickle.load(open(model_path, 'rb'))

        start_time = datetime.now()
        y_pred = model.predict(X_open_mc)
        time_taken = datetime.now() - start_time

        mae = mean_absolute_error(y_open_mc, y_pred)
        mse = mean_squared_error(y_open_mc, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_open_mc, y_pred)

        model_eval_results.loc[model_eval_results['model_id'] == model_id, 'openmc_time_taken'] = time_taken.total_seconds()
        model_eval_results.loc[model_eval_results['model_id'] == model_id, 'openmc_mae'] = mae
        model_eval_results.loc[model_eval_results['model_id'] == model_id, 'openmc_mse'] = mse
        model_eval_results.loc[model_eval_results['model_id'] == model_id, 'openmc_rmse'] = rmse
        model_eval_results.loc[model_eval_results['model_id'] == model_id, 'openmc_r2'] = r2

        model_eval_results.to_csv(model_eval_results_path, index=False)

        print(f'{model_id} tested. MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')