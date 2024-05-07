import pickle
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression

models_path = 'data_store/modelling/models'


def averaging_ensemble(model_ids, X_train, y_train):
    models = []
    for model_id in model_ids:
        with open(f'{models_path}/{model_id}.pkl', 'rb') as file:
            model = pickle.load(file)
            models.append((model_id, model))

    ensemble = VotingRegressor(estimators=models, n_jobs=-1)
    ensemble.fit(X_train, y_train)
    pickle.dump(ensemble, open(f'{models_path}/averaged_ensemble.pkl', 'wb'))


def stacked_ensemble(model_ids, X_train, y_train):
    models = []
    for model_id in model_ids:
        with open(f'{models_path}/{model_id}.pkl', 'rb') as file:
            model = pickle.load(file)
            models.append((model_id, model))

    ensemble = StackingRegressor(estimators=models, final_estimator=LinearRegression(), n_jobs=-1)
    ensemble.fit(X_train, y_train)
    pickle.dump(ensemble, open(f'{models_path}/stacked_ensemble.pkl', 'wb'))
