from sklearn.model_selection import cross_validate
import pandas as pd


import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_validate


def evaluate_pipelines_timeaware(
    pipe_dict,
    X,
    y,
    n_splits=5,
    scoring=("r2", "neg_mean_squared_error", "neg_mean_absolute_error"),
    test_size=None,
    max_train_size=None,
):
    """
    Evaluate multiple pipelines using expanding‐window (time‐aware) CV.

    Parameters
    ----------
    pipe_dict : dict
        mapping of {name: estimator or Pipeline}
    X : DataFrame or array-like
        feature matrix, ordered in time
    y : array-like
        target vector, aligned with X
    n_splits : int, default=5
        number of splits for TimeSeriesSplit
    scoring : tuple of str
        scoring metrics (same conventions as cross_validate)
    test_size : int or None
        number of samples to include in each test fold (None -> equal splits)
    max_train_size : int or None
        max size of the training window (older samples dropped once exceeded)

    Returns
    -------
    DataFrame
        columns: model, train_{metric}, val_{metric} for each metric
    """
    tscv = TimeSeriesSplit(
        n_splits=n_splits, test_size=test_size, max_train_size=max_train_size
    )

    records = []
    for name, pipe in pipe_dict.items():
        cvres = cross_validate(
            pipe, X, y, cv=tscv, scoring=scoring, return_train_score=True, n_jobs=-1
        )
        rec = {"model": name}
        for m in scoring:
            tr = cvres[f"train_{m}"]
            te = cvres[f"test_{m}"]
            mean_tr, mean_te = tr.mean(), te.mean()
            if m.startswith("neg_"):
                nm = m[4:]
                rec[f"train_{nm}"] = -mean_tr
                rec[f"val_{nm}"] = -mean_te
            else:
                rec[f"train_{m}"] = mean_tr
                rec[f"val_{m}"] = mean_te
        records.append(rec)

    return pd.DataFrame.from_records(records)


def evaluate_pipelines(
    pipe_dict,
    X,
    y,
    cv=5,
    scoring=("r2", "neg_mean_squared_error", "neg_mean_absolute_error"),
):
    """
    Evaluate multiple pipelines using cross-validation and return a DataFrame of scores.

    Parameters
    ----------
    pipe_dict : dict
        Mapping of {name: sklearn Pipeline or estimator}
    X : array-like or DataFrame
        Feature matrix
    y : array-like
        Target vector
    cv : int, default=5
        Number of cross-validation folds
    scoring : tuple of str, default=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error')
        Scoring metrics to compute. Negative metrics will be returned as positive values.

    Returns
    -------
    DataFrame
        Columns: model, train_{metric}, val_{metric} for each metric
    """
    records = []
    for name, pipe in pipe_dict.items():
        cvres = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1
        )
        rec = {"model": name}
        for m in scoring:
            # obtain training and test (validation) arrays
            train_key = f"train_{m}"
            test_key = f"test_{m}"
            train_scores = cvres[train_key]
            test_scores = cvres[test_key]
            mean_train = train_scores.mean()
            mean_test = test_scores.mean()
            # if negative metric, flip sign and strip 'neg_'
            if m.startswith("neg_"):
                metric_name = m[4:]
                rec[f"train_{metric_name}"] = -mean_train
                rec[f"val_{metric_name}"] = -mean_test
            else:
                rec[f"train_{m}"] = mean_train
                rec[f"val_{m}"] = mean_test
        records.append(rec)
    return pd.DataFrame.from_records(records)
