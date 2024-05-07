import time
import json

from sklearn.neural_network import MLPRegressor

from utils.modelling import cross_validation as cv


def _update_best_feat_metrics(best_feat_metrics, scores, feature,
                              direction=None, best_features=None, feature_selection_log=None, X=None):
    if direction not in ['forward', 'backward']:
        raise ValueError("direction must be either 'forward' or 'backward'")

    best_feat_metrics['r2'] = scores['r2']
    best_feat_metrics['RMSE'] = scores['rmse']
    best_feat_metrics['MSE'] = scores['mse']
    best_feat_metrics['MAE'] = scores['mae']

    if direction == 'forward':
        if best_features is None:
            raise ValueError("best_features must be provided if direction is 'forward'")

        best_feat_metrics['feature_added'] = feature
        best_feat_metrics['all_features'] = best_features + [feature]

    elif direction == 'backward':
        if feature_selection_log is None:
            raise ValueError("feature_selection_log must be provided if direction is 'backward'")
        if X is None:
            raise ValueError("X must be provided if direction is 'backward'")

        best_feat_metrics['feature_removed'] = feature
        all_removed_features = [log['feature_removed'] for log in feature_selection_log] + [feature] if \
            len(feature_selection_log) > 0 else [feature]
        best_feat_metrics['all_features'] = [col for col in X.columns if col not in all_removed_features]

    return best_feat_metrics


def forwards_feature_selection(train_df, model=None, scoring=None, verbose=True):
    if model is None:
        model = MLPRegressor(random_state=9320)
    if scoring is None:
        scoring = 'r2'

    start_time = time.time()

    X = train_df.drop(columns=['Keff'])
    y = train_df['Keff']

    best_features = []
    feature_selection_log = []
    features_to_check = list(X.columns)

    while len(features_to_check) > 0:
        if verbose:
            print("__________________________________________________________________")
            print(f"SEARCHING FOR FEATURE {len(best_features) + 1} OF {len(X.columns)}")
            print("__________________________________________________________________")

        best_feat_metrics = {'num_of_features': len(best_features) + 1, 'feature_added': None,
                             'r2': -1, 'RMSE': 1, 'MSE':1, 'MAE': 1, 'all_features': []}

        for idx, feature in enumerate(features_to_check):
            if len(best_features) == 0:
                X_temp = X[[feature]]
            else:
                X_temp = X[best_features + [feature]]

            scores = cv.cross_validation(X_temp, y, model=model)

            if scoring == 'r2':
                if scores[scoring] > best_feat_metrics[scoring]:
                    best_feat_metrics = _update_best_feat_metrics(best_feat_metrics, scores, feature,
                                                                  best_features=best_features, direction='forward')
            else:
                if scores[scoring] < best_feat_metrics[scoring]:
                    best_feat_metrics = _update_best_feat_metrics(best_feat_metrics, scores, feature,
                                                                  best_features=best_features, direction='forward')

            if verbose:
                print(f"\rCHECKED FEATURE {idx + 1} OF {len(features_to_check)} | "
                      f"CURRENT BEST {scoring.upper()}: {best_feat_metrics[scoring]:.4f} | "
                      f"USING FEATURE: {best_feat_metrics['feature_added']}", end="")

        if verbose:
            print(f"\n\nNEW FEATURE ADDED: {best_feat_metrics['feature_added']}\n"
                  f"R2: {best_feat_metrics['r2']:.4f}\n"
                  f"RMSE: {best_feat_metrics['RMSE']:.4f}\n"
                  f"MSE: {best_feat_metrics['MSE']:.4f}\n"
                  f"MAE: {best_feat_metrics['MAE']:.4f}\n")

        best_features.append(best_feat_metrics['feature_added'])
        features_to_check.remove(best_feat_metrics['feature_added'])
        feature_selection_log.append(best_feat_metrics)

        with open(f"{model.__class__.__name__}_forward_feature_selection_log.json", 'w') as f:
            json.dump(feature_selection_log, f, indent=4)

    end_time = time.time()
    feature_selection_log.append({'time_taken': end_time - start_time,
                                  'model': model.__class__.__name__})

    with open(f"{model.__class__.__name__}_forward_feature_selection_log.json", 'w') as f:
        json.dump(feature_selection_log, f, indent=4)

    print(f"FORWARDS FEATURE SELECTION COMPLETED IN {(end_time - start_time) / 60} MINUTES")

    return feature_selection_log


def backwards_feature_selection(train_df, model=None, scoring=None, verbose=True):
    if model is None:
        model = MLPRegressor(random_state=9320)
    if scoring is None:
        scoring = 'r2'

    start_time = time.time()

    X = train_df.drop(columns=['Keff'])
    y = train_df['Keff']

    feature_selection_log = []
    remaining_features = list(X.columns)

    while len(remaining_features) > 1:
        if verbose:
            print("__________________________________________________________________")
            print(f"SEARCHING FOR FEATURE TO REMOVE, CURRENTLY {len(remaining_features)} FEATURES")
            print("__________________________________________________________________")

        best_feat_metrics = {'num_of_features': len(remaining_features) - 1, 'feature_removed': None,
                             'r2': -1, 'RMSE': 1, 'MSE': 1, 'MAE': 1, 'all_features': []}

        for idx, feature in enumerate(remaining_features):
            X_temp = X[remaining_features]
            X_temp = X_temp.drop(columns=[feature])

            scores = cv.cross_validation(X_temp, y, model=model)

            if scoring == 'r2':
                if scores[scoring] > best_feat_metrics[scoring]:
                    best_feat_metrics = _update_best_feat_metrics(best_feat_metrics, scores, feature,
                                                                  direction='backward',
                                                                  feature_selection_log=feature_selection_log, X=X)
            else:
                if scores[scoring] < best_feat_metrics[scoring]:
                    best_feat_metrics = _update_best_feat_metrics(best_feat_metrics, scores, feature,
                                                                  direction='backward',
                                                                  feature_selection_log=feature_selection_log, X=X)

            if verbose:
                print(f"\rCHECKED FEATURE {idx + 1} OF {len(remaining_features)} | "
                      f"CURRENT BEST {scoring.upper()}: {best_feat_metrics[scoring]:.4f} | "
                      f"REMOVED FEATURE: {best_feat_metrics['feature_removed']}", end="")

        if verbose:
            print(f"\n\nFEATURE REMOVED: {best_feat_metrics['feature_removed']}\n"
                  f"R2: {best_feat_metrics['r2']:.4f}\n"
                  f"RMSE: {best_feat_metrics['RMSE']:.4f}\n"
                  f"MSE: {best_feat_metrics['MSE']:.4f}\n"
                  f"MAE: {best_feat_metrics['MAE']:.4f}\n")

        remaining_features.remove(best_feat_metrics['feature_removed'])
        feature_selection_log.append(best_feat_metrics)

        with open(f"{model.__class__.__name__}_backward_feature_selection_log.json", 'w') as f:
            json.dump(feature_selection_log, f, indent=4)

    end_time = time.time()
    feature_selection_log.append({'time_taken': end_time - start_time,
                                  'model': model.__class__.__name__})

    with open(f"{model.__class__.__name__}_backward_feature_selection_log.json", 'w') as f:
        json.dump(feature_selection_log, f, indent=4)

    print(f"BACKWARDS FEATURE SELECTION COMPLETED IN {(end_time - start_time) / 60} MINUTES")

    return feature_selection_log
