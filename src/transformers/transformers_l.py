from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        """
        Custom transformer to drop specified columns from a DataFrame.

        Parameters:
        columns_to_drop (list): List of column names to drop.
        """
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        """
        Fit method (no fitting required for this transformer).

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series or None): Target variable (not used).

        Returns:
        self: Returns the transformer instance.
        """
        return self

    def transform(self, X):
        """
        Transform method to drop specified columns.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Transformed DataFrame with specified columns dropped.
        """
        X = X.copy()
        if self.columns_to_drop:
            X.drop(columns=self.columns_to_drop, inplace=True, errors="ignore")
        return X


class CityBasedImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing column values with mean, differentiated by city.

    Requirements:
        - column "start_week_date" is dropped
        - column "city" is encoded into numerical values

    To apply call:
    imputer = CityBasedImputer(city_column='city')
    df_imputed = imputer.fit_transform(df)
    """

    def __init__(self, city_column):
        self.city_column = city_column
        self.city_means = {}

    def fit(self, X, y=None):  # Changed parameter name and added y=None
        """
        Fit method to calculate city-specific means for each column.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series or None): Target variable (not used).

        Returns:
        self: Returns the transformer instance.
        """

        # check if X['city'] has string values
        if X[self.city_column].dtype == "object":
            # Encode the city column to numerical values
            X["city"] = X[self.city_column].map({"sj": 0, "iq": 1})

        # Loop through each city
        for city in X[self.city_column].unique():
            # For each city, compute the mean for each column
            city_data = X[X[self.city_column] == city]
            self.city_means[city] = city_data.mean(numeric_only=True)

        return self  # Return self for method chaining

    def transform(self, X):  # Changed parameter name
        """
        Transform method to impute missing values based on city-specific means.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Transformed DataFrame with missing values imputed.
        """
        # Create a copy of the data to avoid modifying the original
        X_imputed = X.copy()

        # Loop through each city
        for city in X[self.city_column].unique():
            if city in self.city_means:  # Check if city exists in city_means
                # For each city, fill missing values with the computed mean of that city
                city_mask = X_imputed[self.city_column] == city

                for col in self.city_means[city].index:
                    if col in X_imputed.columns:
                        X_imputed.loc[city_mask, col] = X_imputed.loc[
                            city_mask, col
                        ].fillna(self.city_means[city][col])

        return X_imputed

    def fit_transform(self, X, y=None):  # Changed parameter names and added y=None
        """
        Fit and transform method to handle pipelines.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series or None): Target variable (not used).

        Returns:
        pd.DataFrame: Transformed DataFrame with missing values imputed.
        """
        return self.fit(X, y).transform(X)


class CityMapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, city_column="city", mapping={"sj": 0, "iq": 1}):
        self.city_column = city_column
        self.mapping = mapping

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def transform(self, X):
        X = X.copy()  # Create a copy to avoid modifying the original
        if X[self.city_column].dtype == "object":  # Only transform if it's a string
            X[self.city_column] = X[self.city_column].map(self.mapping)
        return X


class RollingAverageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, window, city_column="city"):
        self.columns = columns
        self.window = window
        self.city_column = city_column

    def fit(self, X, y=None):
        # Store column indices if X is a DataFrame
        if hasattr(X, "columns"):
            self.column_indices_ = {
                col: i for i, col in enumerate(X.columns) if col in self.columns
            }
            self.feature_names_ = list(X.columns)
        else:
            # If X is already a numpy array, use the column positions directly
            self.column_indices_ = {
                col: col
                for col in self.columns
                if isinstance(col, int) and col < X.shape[1]
            }
        return self

    def transform(self, X):
        # Convert to DataFrame if it's a numpy array
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            # If input is numpy array, convert to DataFrame
            X_df = pd.DataFrame(
                X,
                columns=(
                    self.feature_names_ if hasattr(self, "feature_names_") else None
                ),
            )
        else:
            X_df = X.copy()

        # Create new rolling average columns, grouped by city
        for col, idx in self.column_indices_.items():
            new_col_name = f"{col}_rolling_avg_{self.window}"

            # Initialize the new column with NaN values
            X_df[new_col_name] = np.nan

            # For each city, compute rolling averages separately
            for city in X_df[self.city_column].unique():
                # Create a city mask
                city_mask = X_df[self.city_column] == city

                # Get indices for this city
                city_indices = X_df[city_mask].index

                # Compute rolling average for this city only
                city_rolling = (
                    X_df.loc[city_mask, col]
                    .rolling(window=self.window, min_periods=1)
                    .mean()
                )

                # Assign the rolling values back to the original DataFrame
                X_df.loc[city_indices, new_col_name] = city_rolling.values

        # Return numpy array if input was numpy array
        if is_numpy:
            return X_df.values
        return X_df

    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if input_features is None and hasattr(self, "feature_names_"):
            input_features = self.feature_names_

        output_features = list(input_features)
        for col in self.columns:
            output_features.append(f"{col}_rolling_avg_{self.window}")

        return np.array(output_features)


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, lag, city_column="city"):
        """
        Transformer to create lagged features for specified columns, grouped by city.

        Parameters:
        - columns (list): List of column names to create lagged features for.
        - lag (int): Number of records in the past to look back.
        - city_column (str): Column name representing the city for grouping.
        """
        self.columns = columns
        self.lag = lag
        self.city_column = city_column

    def fit(self, X, y=None):
        """
        Fit method (no fitting required for this transformer).

        Parameters:
        - X (pd.DataFrame): Input DataFrame.
        - y (pd.Series or None): Target variable (not used).

        Returns:
        - self: Returns the transformer instance.
        """
        return self

    def transform(self, X):
        """
        Transform method to create lagged features.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with lagged features added.
        """
        X = X.copy()  # Create a copy to avoid modifying the original DataFrame

        # Create new lagged columns, grouped by city
        for col in self.columns:
            new_col_name = f"{col}_lag_{self.lag}"

            # Initialize the new column with NaN values
            X[new_col_name] = np.nan

            # For each city, compute lagged values separately
            for city in X[self.city_column].unique():
                # Create a city mask
                city_mask = X[self.city_column] == city

                # Get indices for this city
                city_indices = X[city_mask].index

                # Compute lagged values for this city
                city_lagged = X.loc[city_mask, col].shift(
                    self.lag, fill_value=X.loc[city_mask, col].iloc[0]
                )

                # Assign the lagged values back to the original DataFrame
                X.loc[city_indices, new_col_name] = city_lagged.values

        return X

    def get_feature_names_out(self, input_features=None):
        """
        Return feature names for output features.

        Parameters:
        - input_features (list or None): List of input feature names.

        Returns:
        - np.array: Array of output feature names.
        """
        if input_features is None:
            input_features = self.columns

        output_features = list(input_features)
        for col in self.columns:
            output_features.append(f"{col}_lag_{self.lag}")

        return np.array(output_features)


class YearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, year_col="year", base_year=1990):
        """
        Transformer to normalize the year column starting from a base year.

        Parameters:
        - year_col (str): The name of the year column.
        - base_year (int): The year to start counting from (default is 1990).
        """
        self.year_col = year_col
        self.base_year = base_year

    def fit(self, X, y=None):
        """
        Fit method (no fitting required for this transformer).

        Parameters:
        - X (pd.DataFrame): Input DataFrame.
        - y (pd.Series or None): Target variable (not used).

        Returns:
        - self: Returns the transformer instance.
        """
        return self

    def transform(self, X):
        """
        Transform method to normalize the year column.

        Parameters:
        - X (pd.DataFrame): Input DataFrame.

        Returns:
        - pd.DataFrame: Transformed DataFrame with the year column normalized.
        """
        X = X.copy()
        X[self.year_col + "_transformed"] = X[self.year_col] - self.base_year + 1
        X = X.drop(self.year_col, axis=1)
        return X


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class WeekOfYearFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        week_col="weekofyear",
        target_col="total_cases",
        new_col_name="weekofyear_avg_cases",
    ):
        self.week_col = week_col
        self.target_col = target_col
        self.new_col_name = new_col_name
        self.week_avg_cases = None

    def fit(self, X, y=None):
        """
        Compute the average total_cases for each weekofyear using the training data.
        """
        if y is None:
            raise ValueError(
                "Target variable `y` is required to compute weekofyear averages."
            )

        # Combine X and y to calculate averages
        data = X.copy()
        data[self.target_col] = y[self.target_col]

        # Compute the average total_cases for each weekofyear
        self.week_avg_cases = (
            data.groupby(self.week_col)[self.target_col].mean().to_dict()
        )
        return self

    def transform(self, X):
        """
        Add the weekofyear_avg_cases feature to the dataset.
        """
        X = X.copy()

        # Map the computed averages to the weekofyear column
        X[self.new_col_name] = X[self.week_col].map(self.week_avg_cases)

        # Fill missing values with 0 (in case of unseen weeks)
        X[self.new_col_name].fillna(0, inplace=True)

        return X


class CompositeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that combines related features into composite features
    to reduce dimensionality while preserving signal.
    """

    def __init__(self, drop_original=True):
        self.drop_original = drop_original

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()

        # Temperature composite (average of related temperature readings)
        temp_columns = [
            col
            for col in X.columns
            if "temp" in col and col != "station_diur_temp_rng_c"
        ]
        if len(temp_columns) > 0:
            X_new["composite_temperature"] = X_new[temp_columns].mean(axis=1)

        # Vegetation index composite
        veg_columns = [col for col in X.columns if "ndvi" in col]
        if len(veg_columns) > 0:
            X_new["composite_vegetation"] = X_new[veg_columns].mean(axis=1)

        # Humidity/moisture composite
        humidity_columns = [col for col in X.columns if "humidity" in col]
        if len(humidity_columns) > 0:
            X_new["composite_humidity"] = X_new[humidity_columns].mean(axis=1)

        # Precipitation composite
        precip_columns = [col for col in X.columns if "precip" in col]
        if len(precip_columns) > 0:
            X_new["composite_precipitation"] = X_new[precip_columns].mean(axis=1)

        # Preserve temperature range/variability as its own feature
        if "station_diur_temp_rng_c" in X.columns:
            X_new["temp_variability"] = X_new["station_diur_temp_rng_c"]

        # Create cyclical features for weekofyear
        if "weekofyear" in X.columns:
            X_new["sin_week"] = np.sin(2 * np.pi * X["weekofyear"] / 52)
            X_new["cos_week"] = np.cos(2 * np.pi * X["weekofyear"] / 52)

        # Drop original columns if requested
        if self.drop_original:
            columns_to_keep = ["city", "year", "weekofyear"]
            columns_to_keep.extend(
                [
                    "composite_temperature",
                    "composite_vegetation",
                    "composite_humidity",
                    "composite_precipitation",
                    "temp_variability",
                    "sin_week",
                    "cos_week",
                ]
            )

            X_new = X_new[[col for col in columns_to_keep if col in X_new.columns]]

        return X_new
