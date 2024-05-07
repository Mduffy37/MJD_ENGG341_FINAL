import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


final_model_path = 'data_store/modelling/models/final_model.pkl'


def test_final_model(X_test, y_test):
    model = pickle.load(open(final_model_path, 'rb'))

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100
    r2 = r2_score(y_test, y_pred)

    print(f'Final model tested -> MAE: {mae}, MSE: {mse}, RMSE: {rmse}, RMSPE: {rmspe}, R2: {r2}')

    return y_pred
