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
   Notebook de prueba de Random Forest

6. `train_boosting.ipynb`:
   Notebook de entrenamiento final. Incluye sliding window, evolution of logloss comparada con numero de arboles, evolucion de curvas ROC y Pr y metricas de negocio


## src

1. `utils.py`:
   Funciones & variables que utilizamos en otros .py

2. `prepare_data.py`:
    Script de limpieza de dataset & añadir target label. Basandose en el notebook `v1_dataset.ipynb`.
   Utiliza las tablas zrive_advertiser_withdrawals.parquet, zrive_dim_advertiser.parquet y zrive_fct_monthly_snapshot_advertiser.parquet  para generar una tabla final processed_data.parquet


3. `feature_engineering.py`:
   Añade nuevas features (antigüedad, meses desde último contrato, ratios, media últimos 3 meses, delta últimos 3 meses, etc.

4. `train.py`:
   Entrenamiento del modelo (basandose en train_boosting.ipynb)

5. `inference.py`:
   Carga el modelo entrenado y hace predicciones


## tests

1. `test_feature_engineering.py`:
    Tests para verificar script feature_engineering.py

2. `test_prepare_data.py`:
   Test para verificar script test_prepare_data.py


## Instrucciones

1. `poetry shell`
2. `poetry install`
3. Crear carpeta data y añadir las tablas zrive_advertiser_withdrawals.parquet, zrive_dim_advertiser.parquet y zrive_fct_monthly_snapshot_advertiser.parquet
4. `python3 -m src.prepare_data` genera la tabla  processed_data.parquet
5. `python3 -m src.feature_engineering` genera la tabla full_data.parquet (la que se utiliza para entrenar)


Una vez generado full_data.parquet ya se pueden utilizar los notebooks de entrenamiento y scripts de train & inference. El notebook de entrenamiento final es train_boosting.ipynb.

   
