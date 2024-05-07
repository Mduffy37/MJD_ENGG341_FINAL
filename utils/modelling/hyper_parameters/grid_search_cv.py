from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


def mlp():
    return {
        'model': MLPRegressor(random_state=9320, max_iter=5000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
        }
    }


def rf():
    return {
        'model': RandomForestRegressor(random_state=9320),
        'params': {
            "n_estimators": [100, 500],
            "max_depth": [None, 10, 50],
            "max_features": ['sqrt', 'log2'],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [True, False]
        }
    }


def svr():
    return {
        'model': SVR(),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'degree': [1, 2, 3],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.01, 0.1, 1],
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
        'n_neighbors': [1, 3, 5, 10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 30, 50, 100],
        'p': [1, 2, 3, 4, 5],
        }
    }


def xgb():
    return {
        'model': XGBRegressor(random_state=9320, n_jobs=-1, verbosity=0),
        'params': {
            'n_estimators': [100, 500],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'colsample_bytree': [0.3, 0.7],
            'gamma': [0, 1],
            'reg_alpha': [0, 8],
            'reg_lambda': [0, 8],
            'subsample': [0.3, 0.7],
            'scale_pos_weight': [1, 5]
        }
    }


def ada():
    return {
        'model': AdaBoostRegressor(random_state=9320),
        'params': {
            'n_estimators': [50, 100, 500, 1000],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10],
            'loss': ['linear', 'square', 'exponential'],
        }
    }


def grid_search_optimise(algorithm, X, y, scoring='neg_root_mean_squared_error'):
    algorithm_dists = {
        'mlp': mlp,
        'rf': rf,
        'svr': svr,
        'lr': lr,
        'knn': knn,
        'xgb': xgb,
        'ada': ada,
    }

    algorithm_dist = algorithm_dists[algorithm]()
    model = algorithm_dist['model']
    param_dist = algorithm_dist['params']
    param_dist = {f'model__{key}': value for key, value in param_dist.items()}

    scaler = MinMaxScaler()
    pipeline = Pipeline([('scaler', scaler), ('model', model)])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_dist,
                               cv=cv,
                               scoring=scoring,
                               verbose=2)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_params = {key.split('__')[1]: value for key, value in best_params.items()}

    return best_params
