# CHURN PREDICTION MODEL

Proyecto de predicción de Churn.


Estructura del proyecto:

├── data
│   ├── zrive_advertiser_withdrawals.parquet
│   ├── zrive_dim_advertiser.parquet
│   └── zrive_fct_monthly_snapshot_advertiser.parquet
├── models
├── notebooks
│   ├── eda.ipynb
│   ├── train_boosting.ipynb
│   ├── train_optimized.ipynb
│   ├── training_v2.ipynb
│   ├── v1_dataset.ipynb
│   └── v3_feature_engineering.ipynb
├── poetry.lock
├── predictions
├── pyproject.toml
├── README.md
├── setup.cfg
├── src
│   ├── feature_engineering.py
│   ├── inference.py
│   ├── prepare_data.py
│   ├── train.py
│   └── utils.py
└── tests
    ├── __init__.py
    ├── test_feature_engineering.py
    └── test_prepare_data.py
