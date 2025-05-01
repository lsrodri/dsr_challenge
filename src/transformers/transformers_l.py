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
            X.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')
        return X

class CityBasedImputer(BaseEstimator, TransformerMixin):
    '''
    Impute missing column values with mean, differentiated by city.

    Requirements:
        - column "start_week_date" is dropped
        - column "city" is encoded into numerical values

    To apply call:
    imputer = CityBasedImputer(city_column='city')
    df_imputed = imputer.fit_transform(df)
    '''
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

        #check if X['city'] has string values
        if X[self.city_column].dtype == 'object':
            # Encode the city column to numerical values
            X['city'] = X[self.city_column].map({'sj': 0, 'iq': 1})

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
                        X_imputed.loc[city_mask, col] = X_imputed.loc[city_mask, col].fillna(self.city_means[city][col])
        
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
    def __init__(self, city_column='city', mapping={'sj': 0, 'iq': 1}):
        self.city_column = city_column
        self.mapping = mapping
        
    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self
    
    def transform(self, X):
        X = X.copy()  # Create a copy to avoid modifying the original
        if X[self.city_column].dtype == 'object':  # Only transform if it's a string
            X[self.city_column] = X[self.city_column].map(self.mapping)
        return X
    
class RollingAverageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, window, city_column='city'):
        self.columns = columns
        self.window = window
        self.city_column = city_column
        
    def fit(self, X, y=None):
        # Store column indices if X is a DataFrame
        if hasattr(X, 'columns'):
            self.column_indices_ = {col: i for i, col in enumerate(X.columns) 
                                   if col in self.columns}
            self.feature_names_ = list(X.columns)
        else:
            # If X is already a numpy array, use the column positions directly
            self.column_indices_ = {col: col for col in self.columns 
                                   if isinstance(col, int) and col < X.shape[1]}
        return self
    
    def transform(self, X):
        # Convert to DataFrame if it's a numpy array
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            # If input is numpy array, convert to DataFrame
            X_df = pd.DataFrame(X, columns=self.feature_names_ if hasattr(self, 'feature_names_') else None)
        else:
            X_df = X.copy()
        
        # Create new rolling average columns, grouped by city
        for col, idx in self.column_indices_.items():
            new_col_name = f'{col}_rolling_avg_{self.window}'
            
            # Initialize the new column with NaN values
            X_df[new_col_name] = np.nan
            
            # For each city, compute rolling averages separately
            for city in X_df[self.city_column].unique():
                # Create a city mask
                city_mask = X_df[self.city_column] == city
                
                # Get indices for this city
                city_indices = X_df[city_mask].index
                
                # Compute rolling average for this city only
                city_rolling = X_df.loc[city_mask, col].rolling(
                    window=self.window, min_periods=1).mean()
                
                # Assign the rolling values back to the original DataFrame
                X_df.loc[city_indices, new_col_name] = city_rolling.values
        
        # Return numpy array if input was numpy array
        if is_numpy:
            return X_df.values
        return X_df
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if input_features is None and hasattr(self, 'feature_names_'):
            input_features = self.feature_names_
        
        output_features = list(input_features)
        for col in self.columns:
            output_features.append(f'{col}_rolling_avg_{self.window}')
        
        return np.array(output_features)