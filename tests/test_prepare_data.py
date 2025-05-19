from datetime import datetime
import pytest
import pandas as pd
from pandas import Period
from pandas.testing import assert_frame_equal, assert_series_equal
from src.prepare_data import (
    add_churn,
    add_churn_from_advertiser_data,
    add_churn_target,
    add_predict_month,
    remove_activity_after_first_churn,
    remove_inactive_periods_without_contract,
    remove_incomplete_users
)


def test_add_churn_basic_case():
    data = {
        'withdrawal_type': ['TOTAL', 'TOTAL', 'PARCIAL', 'TOTAL'],
        'withdrawal_status': ['Aprobada', 'Denegada', 'Aprobada', 'Aprobada'],
        'withdrawal_reason': ['Otra razón', 'Cambio de Contrato', 'Upselling', 'Motivo de churn']
    }
    df = pd.DataFrame(data)
    df = add_churn(df)

    expected_churn = [1, 0, 0, 1]
    assert list(df['churn']) == expected_churn


def test_add_churn_excluded_reasons():
    excluded_reasons = [
        'Upselling-cambio de contrato',
        'Cambio a Bundle Online',
        'Cambio de Contrato/propuesta/producto'
    ]

    data = {
        'withdrawal_type': ['TOTAL'] * 3,
        'withdrawal_status': ['Aprobada'] * 3,
        'withdrawal_reason': excluded_reasons
    }
    df = pd.DataFrame(data)
    df = add_churn(df)

    assert df['churn'].sum() == 0


def test_add_churn_denied_status():
    data = {
        'withdrawal_type': ['TOTAL', 'TOTAL'],
        'withdrawal_status': ['Denegada', 'Aprobada'],
        'withdrawal_reason': ['Razón cualquiera', 'Otra razón']
    }
    df = pd.DataFrame(data)
    df = add_churn(df)

    assert list(df['churn']) == [0, 1]


def test_add_churn_partial_withdrawal():
    data = {
        'withdrawal_type': ['PARCIAL', 'TOTAL'],
        'withdrawal_status': ['Aprobada', 'Aprobada'],
        'withdrawal_reason': ['Razón', 'Razón']
    }
    df = pd.DataFrame(data)
    df = add_churn(df)

    assert list(df['churn']) == [0, 1]


def test_add_churn_empty_dataframe():
    df = pd.DataFrame(columns=['withdrawal_type', 'withdrawal_status', 'withdrawal_reason'])
    df = add_churn(df)

    assert 'churn' in df.columns
    assert len(df) == 0


def test_add_churn_missing_columns():
    df = pd.DataFrame({'some_column': [1, 2, 3]})

    with pytest.raises(KeyError):
        add_churn(df)


def test_add_churn_with_nan_values():
    data = {
        'withdrawal_type': ['TOTAL', None, 'TOTAL'],
        'withdrawal_status': [None, 'Aprobada', 'Aprobada'],
        'withdrawal_reason': ['Razón', 'Cambio de Contrato', None]
    }
    df = pd.DataFrame(data)
    df = add_churn(df)

    assert list(df['churn']) == [0, 0, 0]


def test_add_predict_month_basic():
    df = pd.DataFrame({
        'withdrawal_month': [Period('2023-01', 'M'),
                             Period('2023-02', 'M'),
                             Period('2023-03', 'M')]
    })
    result = add_predict_month(df)
    expected = pd.Series([Period('2022-12', 'M'),
                          Period('2023-01', 'M'),
                          Period('2023-02', 'M')],
                         name='predict_month')
    assert_series_equal(result['predict_month'], expected)


def test_add_predict_month_custom_column_names():
    df = pd.DataFrame({
        'my_month': [202405, 202406]
    })
    expected = pd.DataFrame({
        'my_month': [202405, 202406],
        'my_predict': [202404, 202405]
    })

    result = add_predict_month(df.copy(), predict_col='my_predict', withdrawal_col='my_month')
    assert_frame_equal(result, expected)


def test_add_churn_target_merges_correctly():
    df_monthly = pd.DataFrame({
        'advertiser_zrive_id': [1, 2, 3],
        'month_period': [202401, 202402, 202403]
    })
    df_withdrawals = pd.DataFrame({
        'advertiser_zrive_id': [1, 3],
        'predict_month': [202401, 202403],
        'churn': [1, 1]
    })

    expected = pd.DataFrame({
        'advertiser_zrive_id': [1, 2, 3],
        'month_period': [202401, 202402, 202403],
        'churn': [1.0, 0.0, 1.0]
    })

    result = add_churn_target(df_monthly.copy(), df_withdrawals.copy())
    assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))


def test_remove_activity_after_first_churn_removes_basic():
    df = pd.DataFrame({
        'advertiser_zrive_id': [1, 1, 1, 2, 2, 3],
        'period_int': [202401, 202402, 202403, 202401, 202402, 202401],
        'churn': [0, 1, 0, 0, 0, 1]
    })

    expected = pd.DataFrame({
        'advertiser_zrive_id': [1, 1, 2, 2, 3],
        'period_int': [202401, 202402, 202401, 202402, 202401],
        'churn': [0, 1, 0, 0, 1]
    })

    result = remove_activity_after_first_churn(df.copy())
    assert_frame_equal(
        result.sort_index(axis=1).reset_index(drop=True),
        expected.sort_index(axis=1).reset_index(drop=True)
    )


def test_add_churn_from_advertiser_data_basic():
    df_target = pd.DataFrame({
        'advertiser_zrive_id': [1, 1],
        'period_int': [202401, 202402],
        'churn': [0, 0]
    })

    df_advertiser = pd.DataFrame({
        'advertiser_zrive_id': [1],
        'contrato_churn_date': [datetime(2024, 3, 5)]
    })

    result = add_churn_from_advertiser_data(df_target.copy(), df_advertiser.copy())

    expected = pd.DataFrame({
        'advertiser_zrive_id': [1, 1],
        'period_int': [202401, 202402],
        'churn': [0, 1]
    })

    assert_frame_equal(
        result.sort_index(axis=1).reset_index(drop=True),
        expected.sort_index(axis=1).reset_index(drop=True)
    )


def test_add_churn_from_advertiser_data_with_no_activity_previous_month():
    df_target = pd.DataFrame({
        'advertiser_zrive_id': [3],
        'period_int': [202401],
        'churn': [0]
    })

    df_advertiser = pd.DataFrame({
        'advertiser_zrive_id': [3],
        'contrato_churn_date': [datetime(2024, 4, 1)]
    })

    result = add_churn_from_advertiser_data(df_target.copy(), df_advertiser.copy())

    assert_frame_equal(
        result.sort_index(axis=1).reset_index(drop=True),
        df_target.sort_index(axis=1).reset_index(drop=True)
    )


def test_remove_incomplete_users_basic():
    df = pd.DataFrame({
        'advertiser_zrive_id': [1, 2],
        'period_int': [202403, 202403],
        'churn': [0, 1]
    })

    result = remove_incomplete_users(df)

    assert set(result['advertiser_zrive_id']) == {1, 2}


def test_remove_incomplete_users_with_incomplete_user():
    df = pd.DataFrame({
        'advertiser_zrive_id': [1, 2],
        'period_int': [202401, 202402],
        'churn': [0, 0]
    })

    result = remove_incomplete_users(df, latest_period=202403)

    assert 1 not in result['advertiser_zrive_id'].values


def test_remove_inactive_periods_without_contract_without_contract_and_ads():
    df = pd.DataFrame({
        'has_active_contract': [False, True],
        'monthly_published_ads': [0, 5]
    })

    result = remove_inactive_periods_without_contract(df)

    assert len(result) == 1


def test_remove_inactive_periods_without_contract_but_with_ads():
    df = pd.DataFrame({
        'has_active_contract': [False],
        'monthly_published_ads': [3]
    })

    result = remove_inactive_periods_without_contract(df)

    assert len(result) == 1


def test_remove_inactive_periods_with_contract_but_without_ads():
    df = pd.DataFrame({
        'has_active_contract': [True],
        'monthly_published_ads': [0]
    })

    result = remove_inactive_periods_without_contract(df)

    assert len(result) == 1
