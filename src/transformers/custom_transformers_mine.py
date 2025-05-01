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
            self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
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


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Cap outliers using IQR or Z-score by setting extreme values to the boundary thresholds.

    Parameters
    ----------
    columns : str or list of str or None
        Which columns to consider. If None, does nothing.
    remove : bool
        If False, returns X unchanged.
    method : {'iqr', 'zscore'}
    threshold : float
    """

    def __init__(
        self,
        columns=None,
        remove: bool = True,
        method: str = "iqr",
        threshold: float = 1.5,
    ):
        self.columns = columns
        self.remove = remove
        self.method = method
        self.threshold = threshold

        # will be populated in fit()
        self.quartiles_ = {}
        self.means_ = {}
        self.stds_ = {}

    def get_params(self, deep=True):
        # ensure clone can retrieve init parameters
        return {
            "columns": self.columns,
            "remove": self.remove,
            "method": self.method,
            "threshold": self.threshold,
        }

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        method = self.method.lower()

        # build list of columns to process, filtering out any missing ones
        if self.columns is None:
            self._cols = []
        else:
            cols = (
                [self.columns] if isinstance(self.columns, str) else list(self.columns)
            )
            self._cols = [c for c in cols if c in X.columns]
            missing = set(cols) - set(self._cols)
            if missing:
                print(f"OutlierRemover: skipping missing columns {missing!r}")

        for col in self._cols:
            if method == "iqr":
                Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                self.quartiles_[col] = (Q1, Q3)
            elif method == "zscore":
                self.means_[col] = X[col].mean()
                self.stds_[col] = X[col].std(ddof=0)
            else:
                raise ValueError("method must be 'iqr' or 'zscore'")
        return self

    def transform(self, X: pd.DataFrame):
        # if disabled or no valid cols, just return a copy
        if not self.remove or not getattr(self, "_cols", None):
            return X.copy()

        X = X.copy()

        for col in self._cols:
            if self.method.lower() == "iqr":
                Q1, Q3 = self.quartiles_[col]
                IQR = Q3 - Q1
                lower, upper = Q1 - self.threshold * IQR, Q3 + self.threshold * IQR
            else:  # zscore
                m, s = self.means_[col], self.stds_[col]
                lower, upper = m - self.threshold * s, m + self.threshold * s

            # cap values to the computed bounds
            X[col] = X[col].clip(lower=lower, upper=upper)

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


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TempHumidityLagTransformer(BaseEstimator, TransformerMixin):
    """
    1) Compute:
         temp_k        = (min_temp + max_temp) / 2
         temp_humidity = temp_k * (relative_humidity_percent / 100)
    2) Add lagged columns of temp_humidity by each lag in `lags`, grouping by city.

    Parameters
    ----------
    tmin_col : str
        Name of the min‐temp column (Kelvin).
    tmax_col : str
        Name of the max‐temp column (Kelvin).
    rh_col : str
        Name of the relative humidity (%) column.
    out_temp : str
        Name for the computed temp_k column.
    out_inter : str
        Name for the computed interaction (temp_humidity) column.
    lags : list of int
        Number of periods to lag the interaction by.
    city_col : str
        Name of the city column to group by for lagging.
    sort_cols : list of str
        Columns that define time ordering within each city (e.g. ['year','weekofyear']).
    fill_value : scalar or None
        Value to fill for missing lags. If None, leaves NaN.
    """

    def __init__(
        self,
        tmin_col="reanalysis_min_air_temp_k",
        tmax_col="reanalysis_max_air_temp_k",
        rh_col="reanalysis_relative_humidity_percent",
        out_temp="temp_k",
        out_inter="temp_humidity",
        lags=(1,),  # tuple default
        city_col="city",
        sort_cols=("year", "weekofyear"),
        fill_value=None,
    ):
        self.tmin_col = tmin_col
        self.tmax_col = tmax_col
        self.rh_col = rh_col
        self.out_temp = out_temp
        self.out_inter = out_inter
        self.lags = lags  # ← keep as the exact tuple/list given
        self.city_col = city_col
        self.sort_cols = sort_cols  # ← keep as the exact tuple/list
        self.fill_value = fill_value

    def fit(self, X, y=None):
        # nothing to learn
        return self

    def transform(self, X):
        X = X.copy()

        # 1) compute temp_k
        X[self.out_temp] = (X[self.tmin_col] + X[self.tmax_col]) / 2.0

        # 2) compute interaction
        X[self.out_inter] = X[self.out_temp] * (X[self.rh_col] / 100.0)

        # 3) sort within each city
        # make sure we don't lose index
        original_index = X.index
        X = X.sort_values(self.sort_cols + [self.city_col])

        # 4) group & shift for each lag
        grouped = X.groupby(self.city_col)[self.out_inter]
        for lag in self.lags:
            col_name = f"{self.out_inter}_lag_{lag}"
            shifted = grouped.shift(lag)
            if self.fill_value is not None:
                shifted = shifted.fillna(self.fill_value)
            X[col_name] = shifted

        # 5) restore original ordering
        return X.loc[original_index].copy()


class LaggedNDVIRainInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Compute:
      ndvi_mean = mean of the four NDVI quadrants
      rain_ndvi = precipitation_amt_mm * ndvi_mean
    Then for each lag in `lags`, create:
      rain_ndvi_lag_{lag}
      (optionally ndvi_mean_lag_{lag})

    Parameters
    ----------
    ndvi_cols : tuple of str
        The four NDVI quadrant column names.
    rain_col : str
        The precipitation column name.
    out_ndvi : str
        Name of the new ndvi_mean column.
    out_inter : str
        Name of the new rain_ndvi column.
    lags : list of int
        Which weekly lags to compute.
    city_col : str
        Column to group by when shifting (so lags stay within each city).
    keep_ndvi_lags : bool
        If True, also output ndvi_mean_lag_{lag} columns.
    """

    def __init__(
        self,
        ndvi_cols=("ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"),
        rain_col="precipitation_amt_mm",
        out_ndvi="ndvi_mean",
        out_inter="rain_ndvi",
        lags=(1, 3, 5),  # keep it as tuple or list
        city_col="city",
        keep_ndvi_lags=False,
    ):
        self.ndvi_cols = ndvi_cols
        self.rain_col = rain_col
        self.out_ndvi = out_ndvi
        self.out_inter = out_inter
        self.lags = lags  # store exactly what was passed
        self.city_col = city_col
        self.keep_ndvi_lags = keep_ndvi_lags

    def fit(self, X, y=None):
        # nothing to learn
        return self

    def transform(self, X):
        X = X.copy()
        # 1) compute base features
        X[self.out_ndvi] = X[list(self.ndvi_cols)].mean(axis=1)
        X[self.out_inter] = X[self.rain_col] * X[self.out_ndvi]

        # 2) sort so shifts make sense (by city, then time)
        if "year" in X.columns and "weekofyear" in X.columns:
            X = X.sort_values([self.city_col, "year", "weekofyear"])
        else:
            X = X.sort_index()

        # 3) group by city and create lags
        grouped = X.groupby(self.city_col)

        for lag in self.lags:
            lag_ndvi = f"{self.out_ndvi}_lag_{lag}"
            lag_inter = f"{self.out_inter}_lag_{lag}"
            if self.keep_ndvi_lags:
                X[lag_ndvi] = grouped[self.out_ndvi].shift(lag)
            X[lag_inter] = grouped[self.out_inter].shift(lag)

        # 4) restore original order if desired
        X = X.sort_index()

        return X


from sklearn.base import BaseEstimator, TransformerMixin


class LaggedHeatHumidityStressIndexTransformer(BaseEstimator, TransformerMixin):
    """
    1) Compute:
         diurnal_range = tmax_col - tmin_col
         heat_stress   = diurnal_range * (rh_col / 100)
    2) For each lag in `lags`, add:
         heat_stress_lag_{lag}
         (optionally diurnal_range_lag_{lag})

    Parameters
    ----------
    tmin_col : str
        Name of the min‐temp column (Kelvin).
    tmax_col : str
        Name of the max‐temp column (Kelvin).
    rh_col : str
        Name of the relative humidity (%) column.
    out_range : str
        Name of the computed diurnal range column.
    out_index : str
        Name of the computed heat stress column.
    lags : tuple of int
        The weekly lags to generate.
    city_col : str
        Column to group by when shifting (so lags stay within each city).
    keep_range_lags : bool
        If True, also output diurnal_range_lag_{lag} columns.
    """

    def __init__(
        self,
        tmin_col: str = "reanalysis_min_air_temp_k",
        tmax_col: str = "reanalysis_max_air_temp_k",
        rh_col: str = "reanalysis_relative_humidity_percent",
        out_range: str = "diurnal_range",
        out_index: str = "heat_stress",
        lags: tuple[int, ...] = (1, 3, 5),
        city_col: str = "city",
        keep_range_lags: bool = False,
    ):
        # store exactly as passed for clone compatibility
        self.tmin_col = tmin_col
        self.tmax_col = tmax_col
        self.rh_col = rh_col
        self.out_range = out_range
        self.out_index = out_index
        self.lags = lags
        self.city_col = city_col
        self.keep_range_lags = keep_range_lags

    def fit(self, X, y=None):
        # nothing to learn
        return self

    def transform(self, X):
        X = X.copy()

        # 1) compute base features
        X[self.out_range] = X[self.tmax_col] - X[self.tmin_col]
        X[self.out_index] = X[self.out_range] * (X[self.rh_col] / 100.0)

        # 2) ensure proper ordering before lagging
        if all(c in X.columns for c in ("year", "weekofyear")):
            X = X.sort_values([self.city_col, "year", "weekofyear"])
        else:
            X = X.sort_index()

        # 3) group and shift
        grouped_range = X.groupby(self.city_col)[self.out_range]
        grouped_index = X.groupby(self.city_col)[self.out_index]

        for lag in self.lags:
            range_lag_col = f"{self.out_range}_lag_{lag}"
            index_lag_col = f"{self.out_index}_lag_{lag}"

            if self.keep_range_lags:
                X[range_lag_col] = grouped_range.shift(lag)
            X[index_lag_col] = grouped_index.shift(lag)

        # 4) restore original row order
        return X.sort_index()
