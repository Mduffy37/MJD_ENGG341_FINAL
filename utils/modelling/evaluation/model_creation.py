import pandas as pd
import os
import json
import pickle
from datetime import datetime


def train_and_save_models(X_train, y_train, default_algos: dict):
    hp_results_path = 'data_store/modelling/hyper_parameter_tuning/hyper_parameter_tuning_results.csv'
    hp_results = pd.read_csv(hp_results_path)

    model_eval_results_path = 'data_store/modelling/model_performance/model_evaluation_results.csv'
    try:
        model_eval_results = pd.read_csv(model_eval_results_path)
    except FileNotFoundError:
        model_eval_results = pd.DataFrame(columns=['model_id', 'model_name', 'training_time',
                                                   'cv_mae', 'cv_mse', 'cv_rmse', 'cv_r2',
                                                   'test_mae', 'test_mse', 'test_rmse', 'test_r2', ])

    for model_id, model in default_algos.items():
        print(f'Training {model_id}...')
        model_path = f'data_store/modelling/models/{model_id}.pkl'

        if os.path.exists(model_path):
            print(f'{model_id} already exists. Skipping...')
            continue

        hyper_params = hp_results[hp_results['model_id'] == model_id].iloc[0]['optuna_best_params']
        cv_mae = hp_results[hp_results['model_id'] == model_id].iloc[0]['optuna_cv_mae']
        cv_mse = hp_results[hp_results['model_id'] == model_id].iloc[0]['optuna_cv_mse']
        cv_rmse = hp_results[hp_results['model_id'] == model_id].iloc[0]['optuna_cv_rmse']
        cv_r2 = hp_results[hp_results['model_id'] == model_id].iloc[0]['optuna_cv_r2']

        hyper_params = json.loads(hyper_params)

        if model_id not in ['svr', 'lr', 'knn']:
            hyper_params['random_state'] = 9320

        if model_id not in ['lr', 'knn', 'ada']:
            hyper_params['verbose'] = 1

        model = model()
        model.set_params(**hyper_params)
        model_params = model.get_params()

        start_time = datetime.now()
        model.fit(X_train, y_train)
        time_taken = datetime.now() - start_time

        model_eval_results = model_eval_results.append({
            'model_id': model_id,
            'model_name': model.__class__.__name__,
            'training_time': time_taken.total_seconds(),
            'cv_mae': cv_mae,
            'cv_mse': cv_mse,
            'cv_rmse': cv_rmse,
            'cv_r2': cv_r2,
            'test_mae': None,
            'test_mse': None,
            'test_rmse': None,
            'test_r2': None,
        }, ignore_index=True)

        model_eval_results.to_csv(model_eval_results_path, index=False)

        pickle.dump(model, open(model_path, 'wb'))

        with open(f'data_store/modelling/models/{model_id}_params.json', 'w') as f:
            json.dump(model_params, f, indent=4)

        print(f'{model_id} trained and saved to {model_path}')


