
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from utils.modelling import cross_validation as cv


def mlp(trial):
    def suggest_layers():
        depth = trial.suggest_int('depth', 1, 5)
        layers = []
        for i in range(depth):
            layers.append(trial.suggest_int(f'layer_{i}', 1, 100))
        return layers

    return {
        'model': MLPRegressor(random_state=9320, max_iter=5000),
        'params': {
            'hidden_layer_sizes': suggest_layers(),
            'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
            'alpha': trial.suggest_float('alpha', 0.0001, 1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            'tol': trial.suggest_float('tol', 1e-5, 1e-1, log=True),
        }
    }


def rf(trial):
    return {
        'model': RandomForestRegressor(random_state=9320),
        'params': {
                "n_estimators": trial.suggest_int('n_estimators', 100, 1000, step=10),
                "max_depth": trial.suggest_categorical('max_depth', [None] + list(range(1, 101))),
                "max_features": trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                "min_samples_split": trial.suggest_int('min_samples_split', 2, 10),
                "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 4),
                "bootstrap": trial.suggest_categorical('bootstrap', [True, False])
            }
    }


def svr(trial):
    return {
        'model': SVR(),
        'params': {
            'C': trial.suggest_float('C', 0.01, 10, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'degree': trial.suggest_int('degree', 1, 5),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1, log=True),
            'shrinking': trial.suggest_categorical('shrinking', [True, False]),
        }
    }


def lr(trial):
    return {
        'model': LinearRegression(n_jobs=-1),
        'params': {
            'positive': trial.suggest_categorical('positive', [True, False]),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }
    }


def knn(trial):
    return {
        'model': KNeighborsRegressor(n_jobs=-1),
        'params': {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 10),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': trial.suggest_int('leaf_size', 10, 100),
            'p': trial.suggest_int('p', 1, 10),
        }
    }


def xgb(trial):
    return {
        'model': XGBRegressor(random_state=9320, n_jobs=-1, verbosity=0),
        'params': {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=10),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.1, 1),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }
    }


def ada(trial):
    return {
        'model': AdaBoostRegressor(random_state=9320),
        'params': {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=10),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 10, log=True),
            'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
        }
    }


def objective(trial, algorithm, X, y, scoring=None):
    if scoring is None:
        raise ValueError('scoring was not provided')

    algorithm_objectives = {
        'mlp': mlp,
        'rf': rf,
        'svr': svr,
        'lr': lr,
        'knn': knn,
        'xgb': xgb,
        'ada': ada,
    }

    algorithm_objective = algorithm_objectives[algorithm]
    model_info = algorithm_objective(trial)
    model = model_info['model']
    model.set_params(**model_info['params'])

    scores = cv.cross_validation(X, y, model=model)

    if scoring not in scores:
        raise ValueError(f"scoring {scoring} not in scores")
    else:
        return scores[scoring]
