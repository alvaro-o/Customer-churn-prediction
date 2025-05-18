from typing import Tuple, Dict, Any
import pandas as pd
import os
import logging
import joblib
import datetime
import xgboost as xgb
from matplotlib.dates import relativedelta
from src.utils import build_dataframe, FEATURE_COLS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/"))

LABEL_COL = "churn"

LAST_TRAINING_MONTH = datetime.datetime(2024, 11, 1) 


def feature_label_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS], df[LABEL_COL]

def split_train_by_period(
    data_set: pd.DataFrame,
    execution_date: datetime.datetime,
    n_training_months: int = 6,
) -> pd.DataFrame:
    train_start_dt = execution_date - relativedelta(months=n_training_months)

    train_start = pd.Period(train_start_dt.strftime("%Y-%m"), freq="M")
    train_end = pd.Period(execution_date.strftime("%Y-%m"), freq="M")

    train_set = data_set[
        (data_set["month_period"] > train_start) & (data_set["month_period"] <= train_end)
    ]

    logger.info(
        f"Train period: {train_start_dt.strftime('%Y-%m-%d')} to {execution_date.strftime('%Y-%m-%d')} "
        f"({n_training_months} months)"
    )

    unique_months = sorted(train_set["month_period"].unique())
    logger.info(f"Distinct months in training set: {[str(month) for month in unique_months]}")

    return train_set


def save_model(model: Any, model_name: str) -> None:
    if not os.path.exists(OUTPUT_PATH):
        logger.info(f"Creating directory {OUTPUT_PATH}")
        os.makedirs(OUTPUT_PATH)

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    file_path = os.path.join(OUTPUT_PATH, f"{timestamp}_{model_name}.pkl")

    logger.info(f"Saving model to {file_path}")
    joblib.dump(model, file_path)


def train_model(
    train_set: pd.DataFrame,
    n_estimators: int = 150
) -> Tuple[xgb.Booster, Dict[str, Dict[str, Any]]]:
    X_train, y_train = feature_label_split(train_set)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.01,
        "max_depth": 3,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "nthread": 10,
        "random_state": 1
    }

    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train")],
        verbose_eval=False,
        evals_result=evals_result
    )

    return model, evals_result

def train(df: pd.DataFrame) -> None:
    train_set = split_train_by_period(df, LAST_TRAINING_MONTH)

    model, _ = train_model(train_set)

    save_model(model, "xgboost")


def main():
    df = build_dataframe()
    train(df)


if __name__ == "__main__":
    main()
