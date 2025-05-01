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
            self.encoder_ = OneHotEncoder(sparse=False, handle_unknown="ignore")
            reshaped = X[["city"]]
            self.encoder_.fit(reshaped)
            self.categories_ = self.encoder_.categories_[0].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.city is None:
            # one-hot encode with fitted encoder
            encoded = self.encoder_.transform(X[["city"]])
            col_names = [f"city_{cat}" for cat in self.categories_]
            df_enc = pd.DataFrame(encoded, columns=col_names, index=X.index)
            X = pd.concat([X.drop(columns=["city"]), df_enc], axis=1)
            return X.reset_index(drop=True)
        # else filter rows
        return X[X["city"] == self.city].reset_index(drop=True)


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Removes rows containing outliers, detected column-wise by either
    the Inter-Quartile Range (IQR) rule or the Z-score rule.

    Parameters
    ----------
    columns   : str | list[str] | None, default=None
        Column(s) to check.  If None, the transformer is a no-op
        (it returns the data unchanged).
    remove    : bool, default=True
        Whether to drop the outlier rows.  If False, the transformer
        is also a no-op.
    method    : {'iqr', 'zscore'}, default='iqr'
        Outlier detection rule.
    threshold : float, default=1.5
        • IQR  : multiplier of the inter-quartile range
        • Z-score : maximum absolute Z-score
    """

    def __init__(
        self,
        columns: Union[str, List[str], None] = None,
        remove: bool = True,
        method: str = "iqr",
        threshold: float = 1.5,
    ):
        # store parameters exactly as passed (for sklearn.clone)
        self.columns = columns
        self.remove = remove
        self.method = method
        self.threshold = threshold

        # learned statistics
        self.quartiles_: dict = {}
        self.means_: dict = {}
        self.stds_: dict = {}

    # --------------------------------------------------------------------- #
    #                                FIT                                    #
    # --------------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y=None):
        # if no columns specified or removal disabled, nothing to learn
        if self.columns is None or not self.remove:
            return self

        cols = [self.columns] if isinstance(self.columns, str) else self.columns
        method = self.method.lower()

        for col in cols:
            if method == "iqr":
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                self.quartiles_[col] = (q1, q3)
            elif method == "zscore":
                self.means_[col] = X[col].mean()
                self.stds_[col] = X[col].std(ddof=0)
            else:
                raise ValueError("method must be 'iqr' or 'zscore'")

        return self

    # --------------------------------------------------------------------- #
    #                              TRANSFORM                                #
    # --------------------------------------------------------------------- #
    def transform(self, X):
        X = X.copy()
        method = self.method.lower()
        for col in self.columns or []:
            if method == "iqr":
                Q1, Q3 = self.quartiles_[col]
                IQR = Q3 - Q1
                lower, upper = Q1 - self.threshold * IQR, Q3 + self.threshold * IQR
                X[col] = X[col].clip(lower, upper)
            else:  # zscore
                m, s = self.means_[col], self.stds_[col]
                z = (X[col] - m).abs() / s
                # replace extreme with the threshold boundary
                mask = z > self.threshold
                X.loc[mask, col] = (
                    m + np.sign(X.loc[mask, col] - m) * self.threshold * s
                )
        return X


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
