import pandas as pd
from pathlib import Path
from utils import (
    convert_datetime_to_month_period, 
    convert_period_int_to_month_period,
)

SCRIPT_DIR = Path(__file__).parent.resolve()

DATA_PATH = SCRIPT_DIR.parent / "data"

def load_data():
    df_withdrawals = pd.read_parquet(DATA_PATH / "zrive_advertiser_withdrawals.parquet")
    df_advertiser = pd.read_parquet(DATA_PATH / "zrive_dim_advertiser.parquet")
    df_monthly = pd.read_parquet(DATA_PATH / "zrive_fct_montly_snapshot_advertiser.parquet")

    return df_withdrawals, df_advertiser, df_monthly

def preprocess_withdrawals(df_withdrawals):
    df_withdrawals = convert_datetime_to_month_period(df_withdrawals, 'withdrawal_creation_date', 'withdrawal_month', True)
    df_withdrawals = add_predict_month(df_withdrawals)
    df_withdrawals = add_churn(df_withdrawals)

    return df_withdrawals



def main():
    df_withdrawals, df_advertiser, df_monthly = load_data()
    logger.info("All datasets successfully loaded.")

    df_withdrawals = preprocess_withdrawals(df_withdrawals)
    df_monthly = preprocess_monthly(df_monthly)
    logger.info("Withdrawals and monthly data preprocessed.")

    df_target = process_target(df_monthly, df_withdrawals, df_advertiser)
    logger.info("Target data processed.")

    df_target.to_parquet(DATA_PATH / "processed_data.parquet")

if __name__ == "__main__":
    main()