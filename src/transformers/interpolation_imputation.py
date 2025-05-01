from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class InterpolationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method="linear", limit_direction="both"):
        self.method = method
        self.limit_direction = limit_direction
        self.numerical_columns_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.numerical_columns_ = X.select_dtypes(include="number").columns.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        X_transformed = X.copy()

        # Apply interpolation per column, grouped by year, weekofyear, and city
        for col in self.numerical_columns_:
            X_transformed[col] = X_transformed.groupby(["year", "weekofyear", "city"])[
                col
            ].transform(
                lambda group: group.interpolate(
                    method=self.method, limit_direction=self.limit_direction
                )
            )

        return X_transformed
