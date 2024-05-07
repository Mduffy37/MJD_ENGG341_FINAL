import random

from scipy.stats import randint, uniform, rv_discrete
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


class HiddenLayerSizesGenerator:
    @staticmethod
    def rvs(*args, **kwargs):
        depth = random.randint(1, 5)
        layers = [random.randint(1, 100) for _ in range(depth)]
        return tuple(layers)


def mlp():
    return {
        'model': MLPRegressor(random_state=9320, max_iter=5000),
        'params': {
            'hidden_layer_sizes': HiddenLayerSizesGenerator(),
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': uniform(0.0001, 1),
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'tol': uniform(1e-5, 1e-1),
        }
    }


def rf():
    return {
        'model': RandomForestRegressor(random_state=9320),
        'params': {
            "n_estimators": randint(100, 1000),
            "max_depth": [None] + list(range(1, 101)),
            "max_features": ['sqrt', 'log2'],
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 4),
            "bootstrap": [True, False]
        }
    }


def svr():
    return {
        'model': SVR(),
        'params': {
            'C': uniform(0.01, 10),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': randint(1, 5),
            'gamma': ['scale', 'auto'],
            'epsilon': uniform(0.01, 1),
            'shrinking': [True, False],
        }
    }


def lr():
    return {
        'model': LinearRegression(n_jobs=-1),
        'params': {
            'positive': [True, False],
            'fit_intercept': [True, False],
        }
    }


def knn():
    return {
        'model': KNeighborsRegressor(n_jobs=-1),
        'params': {
            'n_neighbors': randint(1, 10),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': randint(10, 100),
            'p': randint(1, 10),
        }
    }


def xgb():
    return {
        'model': XGBRegressor(random_state=9320, n_jobs=-1, verbosity=0),
        'params': {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(1, 10),
            'learning_rate': uniform(0.0001, 1),
            'colsample_bytree': uniform(0.1, 1),
            'colsample_bylevel': uniform(0.1, 1),
            'colsample_bynode': uniform(0.1, 1),
            'gamma': uniform(0, 5),
            'reg_alpha': uniform(0, 10),
            'reg_lambda': uniform(0, 10),
            'min_child_weight': randint(1, 10),
            'max_delta_step': randint(0, 10),
            'subsample': uniform(0.1, 1),
            'scale_pos_weight': uniform(1, 10)
        }
    }


def ada():
    return {
        'model': AdaBoostRegressor(random_state=9320),
        'params': {
            'n_estimators': randint(50, 1000),
            'learning_rate': uniform(0.0001, 10),
            'loss': ['linear', 'square', 'exponential'],
        }
    }


def random_search_optimise(algorithm, X, y,
                           scoring='neg_root_mean_squared_error',):
    algorithm_dists = {
        'mlp': mlp,
        'rf': rf,
        'svr': svr,
        'lr': lr,
        'knn': knn,
        'xgb': xgb,
        'ada': ada,
    }

    if algorithm in ['rf', 'svr']:
        n_iter = 100
    else:
        n_iter = 500

    algorithm_dist = algorithm_dists[algorithm]()
    model = algorithm_dist['model']
    param_dist = algorithm_dist['params']
    param_dist = {f'model__{key}': value for key, value in param_dist.items()}

    scaler = MinMaxScaler()
    pipeline = Pipeline([('scaler', scaler), ('model', model)])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(estimator=pipeline,
                                       param_distributions=param_dist,
                                       n_iter=n_iter,
                                       cv=cv,
                                       scoring=scoring,
                                       verbose=2)
    random_search.fit(X, y)

    best_params = random_search.best_params_
    best_params = {key.split('__')[1]: value for key, value in best_params.items()}

    return best_params



