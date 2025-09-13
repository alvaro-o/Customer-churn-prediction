# CUSTOMER CHURN PREDICTION

This project was completed as part of Zrive School Data Science program in collaboration with a small team of students for Komorebi AI. 
Authors: Alejandro Marcos, Álvaro Ortega, Eduardo Ezponda and Maick Mosquera
Mentor: Guillermo Santamaría 

## Objective
Predict which customers are likely to churn in a subscription-based business using historical account and transaction data.

## Tech Stack
Python | Pandas | scikit-learn | XGBoost | Matplotlib | Seaborn | PyTest | Poetry

## Project Structure

```text
├── data
│   ├── zrive_advertiser_withdrawals.parquet
│   ├── zrive_dim_advertiser.parquet
│   └── zrive_fct_monthly_snapshot_advertiser.parquet
├── models
├── docs
│   └── project_report.pdf
├── notebooks
│   ├── archive
│   ├── eda.ipynb
│   ├── train_boosting.ipynb
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

## Methodology
1. **Exploratory Data Analysis (EDA):** understand data distributions, correlations, and missing values.  
2. **Feature Engineering:** create features like customer tenure, ratios, deltas, rolling averages.  
3. **Model Training:** progressive approach: Logistic Regression → Random Forest → XGBoost. 
4. **Evaluation:** ROC-AUC, Precision-Recall, business metrics.  
5. **Inference:** generate churn predictions on new data.


## Results
- Best model: XGBoost
- ROC-AUC of 0.72 (44% improvement over baseline), and PR-AUC of 0.18
- Business KPI optimization: adjusted threshold of the model to maximize ROI
- Model and insights were presented to the company as part of the course deliverables.


## Usage

1. `poetry shell`
2. `poetry install`
3. Create data folder and add files zrive_advertiser_withdrawals.parquet, zrive_dim_advertiser.parquet and zrive_fct_monthly_snapshot_advertiser.parquet
4. `python3 -m src.prepare_data` generates the file processed_data.parquet
5. `python3 -m src.feature_engineering` generates the file full_data.parquet (used for training)

Once full_data.parquet is generated, training notebooks and traind & inference scripts are ready to use. Final training notebook is train_boosting.ipynb.

## Project Report
The full project report is available in `docs/project_report.pdf`.


   
