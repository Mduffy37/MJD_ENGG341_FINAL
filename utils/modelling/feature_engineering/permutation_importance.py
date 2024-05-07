
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
                             r2_score, max_error, root_mean_squared_error)
from sklearn.inspection import permutation_importance


def get_permutation_importances(X_train, y_train, X_test, y_test):
    columns = X_train.columns

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    mlp = MLPRegressor(random_state=9320, verbose=1)
    mlp.fit(X_train, y_train)

    X_test = scaler.transform(X_test)
    y_test_pred = mlp.predict(X_test)
    results = {
        "model": {
            "algo": str(mlp),
            "params": mlp.get_params()
        },
        "metrics": {
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': root_mean_squared_error(y_test, y_test_pred),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred),
            'Max Error': max_error(y_test, y_test_pred)
        },
        "permutation_importances": {}
    }

    permutation_importances = permutation_importance(mlp, X_test, y_test, n_repeats=20, random_state=9320)
    permutation_importances_df = pd.DataFrame(permutation_importances.importances_mean,
                                              index=columns, columns=['perm_importance'])

    for idx, row in permutation_importances_df.iterrows():
        results['permutation_importances'][f"{idx}"] = row['perm_importance']

    return results

