from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


default_algorithms = {
        'mlp': MLPRegressor,
        'rf': RandomForestRegressor,
        'svr': SVR,
        'lr': LinearRegression,
        'knn': KNeighborsRegressor,
        'xgb': XGBRegressor,
        'ada': AdaBoostRegressor
    }
