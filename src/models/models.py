# models.py
"""
Define a comprehensive suite of regression models.
"""
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    SGDRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

# Third-party gradient boosters
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


def get_models(random_state: int = 42) -> dict:
    """
    Return a dictionary of regression model instances.

    Returns
    -------
    dict
        Keys are short names; values are instantiated regressors.
    """
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=0.1, random_state=random_state),
        "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
        "bayesianridge": BayesianRidge(),
        "sgd": SGDRegressor(max_iter=1000, tol=1e-3, random_state=random_state),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "svr": SVR(C=1.0),
        "kernel_ridge": KernelRidge(alpha=1.0),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),
        "extra_trees": ExtraTreesRegressor(n_estimators=100, random_state=random_state),
        "gb": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        "hist_gb": HistGradientBoostingRegressor(
            max_iter=100, random_state=random_state
        ),
        "gp": GaussianProcessRegressor(),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(100,), max_iter=500, random_state=random_state
        ),
    }

    # Conditionally include third-party boosters
    if XGBRegressor is not None:
        models["xgb"] = XGBRegressor(
            n_estimators=100, random_state=random_state, verbosity=0
        )
    if LGBMRegressor is not None:
        models["lgbm"] = LGBMRegressor(n_estimators=100, random_state=random_state)

    return models
