import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CityWiseRollingMeanImputer(BaseEstimator, TransformerMixin):
    """
    Transformer that imputes missing values in numerical columns using a rolling mean,
    calculated separately for each city as indicated by one-hot encoded city columns.

    Automatically detects numeric and city columns based on column names.
    """

    def __init__(self, window_size=3, min_periods=1):
        self.window_size = window_size
        self.min_periods = min_periods
        self.city_cols = None
        self.numeric_cols = None

    def fit(self, X, y=None):
        self._validate_input(X)

        # Detect city columns (e.g. 'city_iq', 'city_sj')
        self.city_cols = [col for col in X.columns if col.startswith("city_")]
        if not self.city_cols:
            raise ValueError(
                "No one-hot encoded city columns found (e.g., 'city_iq', 'city_sj')"
            )

        # Detect numeric columns, excluding city one-hot columns
        self.numeric_cols = (
            X.select_dtypes(include=["number"])
            .columns.difference(self.city_cols)
            .tolist()
        )
        if not self.numeric_cols:
            raise ValueError("No numeric columns found to impute.")

        return self

    def transform(self, X):
        self._validate_input(X)
        if self.city_cols is None or self.numeric_cols is None:
            raise RuntimeError("Transformer has not been fitted yet.")

        X_transformed = X.copy()

        for city_col in self.city_cols:
            city_mask = X_transformed[city_col] == 1

            if not city_mask.any():
                continue

            for num_col in self.numeric_cols:
                city_series = X_transformed.loc[city_mask, num_col]

                if not city_series.isna().any():
                    continue

                rolling_means = city_series.rolling(
                    window=self.window_size, min_periods=self.min_periods
                ).mean()

                rolling_means = rolling_means.ffill().bfill()
                missing_mask = city_mask & X_transformed[num_col].isna()
                X_transformed.loc[missing_mask, num_col] = rolling_means[missing_mask]

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")


def check_imputation(data):
    """
    Test the CityWiseRollingMeanImputer by comparing missing values before and after imputation.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to test imputation on. Must include one-hot encoded city columns
        named like 'city_iq', 'city_sj' and numeric columns with missing values.

    Returns
    -------
    pandas.DataFrame
        The imputed data
    """
    # Count missing values before imputation
    missing_before = data.isna().sum()
    print("Missing values before imputation:")
    print(missing_before)
    print(f"Total missing: {missing_before.sum()}")

    # Apply imputer
    imputer = CityWiseRollingMeanImputer(window_size=3)
    imputed_data = imputer.fit_transform(data)

    # Count missing values after imputation
    missing_after = imputed_data.isna().sum()
    print("\nMissing values after imputation:")
    print(missing_after)
    print(f"Total missing: {missing_after.sum()}")

    # Calculate improvement
    imputed_count = missing_before.sum() - missing_after.sum()
    print(f"\nTotal values imputed: {imputed_count}")
    if missing_before.sum() > 0:
        percentage = (imputed_count / missing_before.sum()) * 100
        print(f"Percentage imputed: {percentage:.2f}%")

    # Compare columns that still have missing values
    print("\nColumns with remaining missing values:")
    for col in missing_after[missing_after > 0].index:
        before = missing_before[col]
        after = missing_after[col]
        if before > after:
            filled = before - after
            print(f"{col}: {after} missing values remaining (filled {filled} values)")
        else:
            print(f"{col}: {after} missing values (no improvement)")

    return imputed_data
