# CHURN PREDICTION MODEL

Proyecto de predicción de Churn.


Estructura del proyecto:

```text
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

```


## Notebooks

1. `v1_dataset.ipynb`:
    - Como definir churn & casos problematicos

2. `eda.ipynb`:
    - Eda muy basico

3. `v3_feature_engineering.ipynb`:
   - Añadir features: ratios, medias & deltas ultimos 3 meses, antigúedad, etc.
  
4. `training_v2.ipynb`:
   - Primeras pruebas de entrenamiento
  
5. `train_optimized.ipynb`:
   Notebook de prueba de Random Forest incluyendo shap

6. `train_boosting.ipynb`:
   Notebook de entrenamiento final. Incluye sliding window, evolution of logloss comparada con numero de arboles, evolucion de curvas ROC y Pr y metricas de negocio


## Scripts .py

1. `utils.py`:
   Funciones & variables que utilizamos en otros .py

2. `prepare_data.py`:
    Script de limpieza de dataset & añadir target label. Basandose en el notebook `v1_dataset.ipynb`.
   Utiliza las tablas `zrive_advertiser_withdrawals.parquet`, `zrive_dim_advertiser.parquet` y `zrive_fct_monthly_snapshot_advertiser.parquet`  para generar una tabla final `processed_data.parquet`
