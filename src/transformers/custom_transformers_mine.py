from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, List
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

class CitySelector(BaseEstimator, TransformerMixin):
    """
    Selects rows for a given city, or one-hot–encodes 'city' when city=None.

    Parameters
    ----------
    city : str or None, default=None
        - If a code like 'sj' or 'iq', filter to that city.
        - If None, return all rows with 'city' one-hot encoded.
    """
    def __init__(self, city: str = None):
        self.city = city
        self.encoder_ = None
        self.categories_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.city is None:
            # fit OneHotEncoder on all observed cities
            self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            reshaped = X[['city']]
            self.encoder_.fit(reshaped)
            self.categories_ = self.encoder_.categories_[0].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.city is None:
            # one-hot encode with fitted encoder
            encoded = self.encoder_.transform(X[['city']])
            col_names = [f'city_{cat}' for cat in self.categories_]
            df_enc = pd.DataFrame(encoded, columns=col_names, index=X.index)
            X = pd.concat([X.drop(columns=['city']), df_enc], axis=1)
            return X.reset_index(drop=True)
        # else filter rows
        return X[X['city'] == self.city].reset_index(drop=True)
    


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    A custom transformer to remove outliers from specified columns using IQR or Z-score methods.

    Parameters
    ----------
    columns : Union[str, List[str]]
        Column name or list of column names to check for outliers.
    remove : bool, default=True
        Whether to remove detected outliers. If False, returns the original dataset.
    method : str, default='iqr'
        Detection method:
          - 'iqr': Interquartile Range method
          - 'zscore': Z-score (standard score) method
    threshold : float, default=1.5
        Outlier cutoff threshold:
          - For 'iqr', multiplier of IQR beyond Q1 and Q3.
          - For 'zscore', maximum absolute Z-score.

    Attributes
    ----------
    quartiles_ : dict
        Q1 and Q3 values for each column when using IQR.
    means_ : dict
        Mean values for each column when using Z-score.
    stds_ : dict
        Standard deviation values for each column when using Z-score.
    """
    def __init__(
        self,
        columns: Union[str, List[str]],
        remove: bool = True,
        method: str = 'iqr',
        threshold: float = 1.5
    ):
        self.columns = [columns] if isinstance(columns, str) else columns
        self.remove = remove
        self.method = method.lower()
        self.threshold = threshold
        self.quartiles_: dict = {}
        self.means_: dict = {}
        self.stds_: dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn parameters for outlier detection.
        """
        X = X.copy()
        for col in self.columns:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                self.quartiles_[col] = (Q1, Q3)
            elif self.method == 'zscore':
                self.means_[col] = X[col].mean()
                self.stds_[col] = X[col].std(ddof=0)
            else:
                raise ValueError("""method must be 'iqr' or 'zscore'""")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows containing outliers based on fitted parameters.
        """
        if not self.remove:
            return X.copy().reset_index(drop=True)

        X = X.copy()
        mask = pd.Series(True, index=X.index)
        for col in self.columns:
            if self.method == 'iqr':
                Q1, Q3 = self.quartiles_[col]
                IQR = Q3 - Q1
                lower, upper = Q1 - self.threshold * IQR, Q3 + self.threshold * IQR
                mask &= X[col].between(lower, upper, inclusive='both')
            else:  # zscore
                mean = self.means_[col]
                std = self.stds_[col]
                z = (X[col] - mean).abs() / std
                mask &= z <= self.threshold

        return X.loc[mask].reset_index(drop=True)


class CyclicTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to encode a cyclical column (like week of year) into sine and cosine features.

    Parameters
    ----------
    column : str
        Name of the column to transform.
    period : int
        The period of the cycle (e.g., 52 for weeks in a year).
    drop_original : bool, default=True
        If True, drop the original column after transformation.
    """
    def __init__(self, column: str, period: int, drop_original: bool = True):
        self.column = column
        self.period = period
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y=None):
        # No fitting necessary
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # compute angle
        angles = 2 * np.pi * X[self.column] / self.period
        # add sine and cosine columns
        X[f"{self.column}_sin"] = np.sin(angles)
        X[f"{self.column}_cos"] = np.cos(angles)
        # optionally drop original
        if self.drop_original:
            X = X.drop(columns=[self.column])
        return X.reset_index(drop=True)