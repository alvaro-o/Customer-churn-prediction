import os
import logging
from joblib import load
import xgboost as xgb
import pandas as pd
from src.train import feature_label_split
from src.utils import OUTPUT_PATH, build_dataframe, evaluate_model, save_predictions

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main() -> None:
    model_name = "20250518-233011_xgboost.pkl"
    model_path = os.path.join(OUTPUT_PATH, model_name)
    model: xgb.Booster = load(model_path)
    logger.info(f"Loaded model {model_name}")

    df: pd.DataFrame = build_dataframe()
    X, y = feature_label_split(df)

    y_pred = model.predict(xgb.DMatrix(X))

    evaluate_model("Inference test", y, y_pred)

    save_predictions(y, y_pred, model_name, df)

if __name__ == "__main__":
    main()
