
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
                             r2_score, max_error, root_mean_squared_error)


def get_feature_importances(X_train, y_train, X_test, y_test):
    columns = X_train.columns
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    rf = RandomForestRegressor(random_state=9320, verbose=1)
    rf.fit(X_train, y_train)

    X_test = scaler.transform(X_test)
    y_test_pred = rf.predict(X_test)
    results = {
        "model": {
            "algo": str(rf),
            "params": rf.get_params()
        },
        "metrics": {
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': root_mean_squared_error(y_test, y_test_pred),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred),
            'Max Error': max_error(y_test, y_test_pred)
        },
        "feature_importances": {}
    }

    importances = rf.feature_importances_
    feature_importances_df = pd.DataFrame(importances, index=columns, columns=['feat_importance'])

    for idx, row in feature_importances_df.iterrows():
        results['feature_importances'][f"{idx}"] = row['feat_importance']

    return results
