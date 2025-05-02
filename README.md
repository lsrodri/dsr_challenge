# DengueAI Prediction Challenge

## Project Overview
This project presents a machine learning solution for the DengueAI competition hosted by DrivenData, predicting dengue fever cases in San Juan, Puerto Rico and Iquitos, Peru using environmental and climate data.

## The Challenge
Dengue fever is a mosquito-borne disease affecting millions globally each year. The competition tasks participants with predicting weekly dengue cases based on environmental variables including temperature, precipitation, vegetation indices, and humidity.

## Solution Approach
### Data Preprocessing Pipeline
The solution implements a custom scikit-learn compatible preprocessing pipeline:

- Feature Selection: Retained relevant environmental features and location identifiers
- Time-Series Processing: Created 3-week rolling averages for key climate variables
- Missing Value Handling: Implemented city-specific imputation for missing data
- Data Normalization: Applied standard scaling to all features

### Custom Transformers
The project includes several custom transformers to handle specific preprocessing needs:

- DropColumnsTransformer: Removes specified columns
- CityMapTransformer: Encodes city names as numerical values
- CityBasedImputer: Imputes missing values using city-specific statistics
- RollingAverageTransformer: Creates temporal features with rolling windows

## Model Architecture
- Algorithm: Random Forest Regressor with 150 estimators
- Training Strategy: 80% of data used for training, 20% for validation
- Full Data Utilization: Final model retrained on complete dataset before competition submission

## Evaluation & Visualizations
The project includes comprehensive model evaluation:

- Actual vs. predicted case comparison
- Time-series analysis of prediction accuracy
- Residual analysis
- City-specific performance tracking
- Spike detection evaluation

## Implementation Details
### Feature Engineering
Key features used in the model include:

- Climate measurements (temperatures, humidity)
- Precipitation data
- Vegetation indices
- Temporal information (year, week)

### Pipeline Architecture

```python
pipeline = Pipeline(steps=[
    ('drop_columns', DropColumnsTransformer()),
    ('city_encoder', CityMapTransformer()),
    ('imputer', CityBasedImputer(city_column='city')),
    ('rolling_avg', RollingAverageTransformer(window=3)),
    ('drop_rolling_columns', DropColumnsTransformer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])
```

## Key Findings
- Time-series features significantly improve prediction accuracy
- City-specific patterns require specialized preprocessing
- Full pipeline approach outperforms separated components
- Model struggles most with predicting large case spikes

## Future Work
- Experiment with different window sizes for rolling averages
- Explore more sophisticated time-series models (LSTM, Prophet)
- Incorporate additional environmental and demographic features
- Implement ensemble approaches combining multiple model types
