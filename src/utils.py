import pandas as pd


def convert_datetime_to_month_period(df, datetime_col, new_col, drop_original=True):
    df[new_col] = pd.to_datetime(df[datetime_col]).dt.to_period('M')
    if drop_original:
        df = df.drop(columns=[datetime_col])
    return df

def convert_period_int_to_month_period(df, period_col='period_int', new_col='month_period'):
    """Convert YYYYMM format to pandas monthly period"""
    df[new_col] = pd.to_datetime(df[period_col].astype(str) + '01', format='%Y%m%d').dt.to_period('M')
    return df