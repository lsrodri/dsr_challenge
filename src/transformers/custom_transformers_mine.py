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
    


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Remove outliers using IQR or Z-score.

    Parameters
    ----------
    columns : str or list of str
    remove  : bool
    method  : {'iqr', 'zscore'}
    threshold : float
    """
    def __init__(self,
                 columns=None,
                 remove: bool = True,
                 method: str = 'iqr',
                 threshold: float = 1.5):
        # store the raw parameters exactly as given
        self.columns = columns
        self.remove = remove
        self.method = method        # ⚠️ do NOT .lower() here
        self.threshold = threshold

        # these will be filled in fit()
        self.quartiles_ = {}
        self.means_ = {}
        self.stds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        # normalize the method value once here
        method = self.method.lower()
        cols = [self.columns] if isinstance(self.columns, str) else self.columns
        for col in cols:
            if method == 'iqr':
                Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                self.quartiles_[col] = (Q1, Q3)
            elif method == 'zscore':
                self.means_[col] = X[col].mean()
                self.stds_[col]  = X[col].std(ddof=0)
            else:
                raise ValueError("method must be 'iqr' or 'zscore'")
        return self

    def transform(self, X: pd.DataFrame):
        if not self.remove:
            return X.copy().reset_index(drop=True)

        X = X.copy()
        mask = pd.Series(True, index=X.index)
        method = self.method.lower()

        for col in self.quartiles_ if method == 'iqr' else self.means_:
            if method == 'iqr':
                Q1, Q3 = self.quartiles_[col]
                IQR = Q3 - Q1
                lower, upper = Q1 - self.threshold * IQR, Q3 + self.threshold * IQR
                mask &= X[col].between(lower, upper)
            else:  # zscore
                mean, std = self.means_[col], self.stds_[col]
                mask &= ((X[col] - mean).abs() / std) <= self.threshold

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