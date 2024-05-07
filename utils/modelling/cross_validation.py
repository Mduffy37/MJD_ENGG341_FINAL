
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


def cross_validation(X, y, model=None, scoring=None):
    if model is None:
        raise ValueError('model was not provided')

    if scoring is None:
        mae_scorer = make_scorer(mean_absolute_error)
        mse_scorer = make_scorer(mean_squared_error)
        rmse_scorer = make_scorer(root_mean_squared_error)
        r2_scorer = make_scorer(r2_score)

        scoring = {'mae': mae_scorer, 'mse': mse_scorer, 'rmse': rmse_scorer, 'r2': r2_scorer}

    scaler = MinMaxScaler()
    pipeline = Pipeline([('scaler', scaler), ('model', model)])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    mean_scores = {}
    for score in scoring.keys():
        mean_scores[score] = cv_scores[f'test_{score}'].mean()

    return mean_scores

