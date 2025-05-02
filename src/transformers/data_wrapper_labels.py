from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class DataFrameWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        if hasattr(self.transformer, "get_feature_names_out"):
            self.feature_names_out_ = self.transformer.get_feature_names_out()
        elif hasattr(X, "columns"):
            self.feature_names_out_ = X.columns
        else:
            self.feature_names_out_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X_t = self.transformer.transform(X)
        return pd.DataFrame(X_t, columns=self.feature_names_out_, index=X.index)
