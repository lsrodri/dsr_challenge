from sklearn.model_selection import cross_validate


def evaluate_pipelines(pipe_dict, X, y, cv=5,
                       scoring=('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error')):
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
        cvres = cross_validate(pipe, X, y, cv=cv,
                               scoring=scoring,
                               return_train_score=True,
                               n_jobs=-1)
        rec = {'model': name}
        for m in scoring:
            # obtain training and test (validation) arrays
            train_key = f'train_{m}'
            test_key = f'test_{m}'
            train_scores = cvres[train_key]
            test_scores = cvres[test_key]
            mean_train = train_scores.mean()
            mean_test = test_scores.mean()
            # if negative metric, flip sign and strip 'neg_'
            if m.startswith('neg_'):
                metric_name = m[4:]
                rec[f'train_{metric_name}'] = -mean_train
                rec[f'val_{metric_name}'] = -mean_test
            else:
                rec[f'train_{m}'] = mean_train
                rec[f'val_{m}'] = mean_test
        records.append(rec)
    return pd.DataFrame.from_records(records)
