import numpy as np
import pandas as pd
from src.feature_engineering import create_agg_stats, create_time_features, engineer_features


def create_time_features_with_activity_before_contract():
    df = pd.DataFrame({
        'advertiser_zrive_id': [2, 2],
        'month_period': [pd.Period('2023-12', freq='M'), pd.Period('2024-02', freq='M')],
        'has_active_contract': [False]
    })

    df_advertiser = pd.DataFrame({
        'advertiser_zrive_id': [2],
        'min_start_contrato_date': ['2024-01-01'],
        'max_start_contrato_nuevo_date': ['2024-03-01'],
        'contrato_churn_date': ['2024-06-01'],
        'province_id': [101],
        'updated_at': ['2024-07-01'],
        'advertiser_province': ['B'],
        'advertiser_group_id': [20]
    })

    result = create_time_features(df, df_advertiser)

    assert result['tenure'].iloc[0] ==  3 # Feb - Dec
    assert result['months_since_last_contract'].iloc[0] == 4  # Dec - Mar
    assert result['has_renewed'].iloc[0] == 1

def create_time_features_with_activity_before_contract_basic():
    df = pd.DataFrame({
        'advertiser_zrive_id': [2],
        'month_period': [pd.Period('2024-07', freq='M')],
        'has_active_contract': [True]
    })

    df_advertiser = pd.DataFrame({
        'advertiser_zrive_id': [2],
        'min_start_contrato_date': ['2024-01-01'],
        'max_start_contrato_nuevo_date': ['2024-03-01'],
        'contrato_churn_date': ['2024-06-01'],
        'province_id': [101],
        'updated_at': ['2024-07-01'],
        'advertiser_province': ['B'],
        'advertiser_group_id': [20]
    })

    result = create_time_features(df, df_advertiser)

    assert result['tenure'].iloc[0] == 6  # Jul - Jan
    assert result['months_since_last_contract'].iloc[0] == 4  # Jul - Mar
    assert result['has_renewed'].iloc[0] == 1

def create_time_features_with_activity_before_contract_no_renewal():
    df = pd.DataFrame({
        'advertiser_zrive_id': [3],
        'month_period': [pd.Period('2025-01', freq='M')],
        'has_active_contract': [False]
    })

    df_advertiser = pd.DataFrame({
        'advertiser_zrive_id': [3],
        'min_start_contrato_date': ['2024-01-01'],
        'max_start_contrato_nuevo_date': [None],
        'contrato_churn_date': ['2024-06-01'],
        'province_id': [102],
        'updated_at': ['2024-12-01'],
        'advertiser_province': ['C'],
        'advertiser_group_id': [30]
    })

    result = create_time_features(df, df_advertiser)

    assert result['tenure'].iloc[0] == 0  # 2025-01 - 2025-01 (actividad sin contrato)
    assert result['months_since_last_contract'].iloc[0] == 0  # se rellena con tenure
    assert result['has_renewed'].iloc[0] == 0

def test_create_agg_stats_all_aggs_with_delta():
    df = pd.DataFrame({
        'advertiser_zrive_id': [1]*3,
        'month_period': pd.period_range('2024-01', periods=3, freq='M'),
        'value': [10, 20, 30]
    })

    result = create_agg_stats(
        df,
        features=['value'],
        months=2,
        agg_funcs=['mean', 'min', 'max', 'std'],
        add_deltas=True
    )

    expected_means = [10, 15, 25]
    expected_deltas = [0, 5, 5]
    expected_mins = [10, 10, 20]
    expected_maxs = [10, 20, 30]
    expected_stds = [0.0, 7.07, 7.07]

    assert result['value_2_months_mean'].tolist() == expected_means
    assert result['value_2_months_mean_delta'].tolist() == expected_deltas
    assert result['value_2_months_min'].tolist() == expected_mins
    assert result['value_2_months_max'].tolist() == expected_maxs

    computed_stds = [round(np.nan_to_num(x, nan=0.0), 2) for x in result['value_2_months_std']]
    assert computed_stds == expected_stds

def test_engineer_features_basic():
    df = pd.DataFrame({
        'advertiser_zrive_id': [1]*3,
        'month_period': pd.period_range('2024-01', periods=3, freq='M'),
        'monthly_leads': [10, 20, 30],
        'monthly_visits': [100, 200, 300],
        'monthly_total_invoice': [1000, 2000, 3000],
        'monthly_avg_ad_price': [50, 60, 70],
        'monthly_published_ads': [5, 10, 15],
        'monthly_contracted_ads': [5, 10, 15],
        'monthly_unique_ads': [0, 1, 2],
        'monthly_unique_published_ads': [0, 1, 2],
        'monthly_premium_ads': [1, 2, 3],
        'monthly_oro_ads': [0, 1, 2],
        'monthly_plata_ads': [0, 1, 2],
        'monthly_destacados_ads': [0, 1, 2],
        'monthly_pepitas_ads': [0, 1, 2],
        'monthly_shows': [100, 100, 100],
        'monthly_unique_ads': [5, 10, 15],
        'has_active_contract': [True]*3
    })

    df_advertiser = pd.DataFrame({
        'advertiser_zrive_id': [1],
        'min_start_contrato_date': ['2023-12-01'],
        'max_start_contrato_nuevo_date': ['2024-01-01'],
        'contrato_churn_date': ['2024-12-01'],
        'province_id': [1],
        'updated_at': ['2024-01-01'],
        'advertiser_province': ['A'],
        'advertiser_group_id': [1]
    })

    result = engineer_features(df, df_advertiser)

    # Verificar que se crearon algunas columnas clave
    expected_cols = [
        'monthly_leads_3_months_mean',
        'monthly_leads_3_months_mean_delta',
        'leads_per_visit_3_months_mean',
        'leads_per_visit_3_months_mean_delta',
    ]

    for col in expected_cols:
        assert col in result.columns, f"{col} not found in result"

    assert result.loc[2, 'monthly_leads_3_months_mean'] == 20
    assert result.loc[2, 'monthly_leads_3_months_mean_delta'] == 10  # 30 - 20
    assert round(result.loc[2, 'leads_per_visit_3_months_mean'], 3) == round((10/100 + 20/200 + 30/300) / 3, 3)






