from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# 1. App init
app = FastAPI(title="Churn Prediction Service")

# 2. Feature schema (from client)
FEATURE_COLUMNS = [
    "cons_gas_12m",
    "cons_last_month",
    "forecast_cons_12m",
    "forecast_discount_energy",
    "forecast_meter_rent_12m",
    "has_gas",
    "nb_prod_act",
    "num_years_antig",
    "pow_max",
    "days_to_end",
    "days_since_modif",
    "days_to_renewal",
    "last_month_vs_avg",
    "gas_to_elec_ratio",
    "margin_ratio",
    "profitability_index",
    "is_high_power",
    "is_low_usage",
    "latest_price_off_peak_fix",
]

# This is what the models were trained on:
# original features + recon_error + is_anomaly
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + ["recon_error", "is_anomaly"]

class CustomerFeatures(BaseModel):
    cons_gas_12m: float
    cons_last_month: float
    forecast_cons_12m: float
    forecast_discount_energy: float
    forecast_meter_rent_12m: float
    has_gas: int
    nb_prod_act: int
    num_years_antig: int
    pow_max: float
    days_to_end: int
    days_since_modif: int
    days_to_renewal: int
    last_month_vs_avg: float
    gas_to_elec_ratio: float
    margin_ratio: float
    profitability_index: float
    is_high_power: int
    is_low_usage: int
    latest_price_off_peak_fix: float

# 3. Load models on startup
rf_model = None
logreg_model = None
svc_model = None

@app.on_event("startup")
def load_models():
    global rf_model, logreg_model, svc_model

    rf_model = joblib.load("models/rf_balanced.pkl")
    logreg_model = joblib.load("models/logreg_balanced.pkl")
    svc_model = joblib.load("models/svc_balanced.pkl")

    print("Classic ML models loaded successfully (expecting recon_error + is_anomaly).")

# 5. Health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Churn API is running (VAE disabled, dummy anomaly features)."}

# 6. Predict endpoint
@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    data_dict = features.dict()
    # DataFrame with only the original 19 features
    df_input = pd.DataFrame([data_dict], columns=FEATURE_COLUMNS)

    # Add dummy anomaly features (since VAE is disabled)
    df_input["recon_error"] = 0.0
    df_input["is_anomaly"] = 0

    # Reorder columns to exactly match model training
    df_model = df_input[MODEL_FEATURE_COLUMNS]

    # Random Forest
    rf_proba = rf_model.predict_proba(df_model)[0, 1]
    rf_pred = int(rf_proba >= 0.5)

    # Logistic Regression
    lr_proba = logreg_model.predict_proba(df_model)[0, 1]
    lr_pred = int(lr_proba >= 0.5)

    # SVC
    svc_proba = svc_model.predict_proba(df_model)[0, 1]
    svc_pred = int(svc_proba >= 0.5)

    return {
        "input": data_dict,
        "random_forest": {
            "churn_proba": float(rf_proba),
            "churn_pred": rf_pred
        },
        "logistic_regression": {
            "churn_proba": float(lr_proba),
            "churn_pred": lr_pred
        },
        "svc": {
            "churn_proba": float(svc_proba),
            "churn_pred": svc_pred
        }
    }
# # app/main.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pickle
# import numpy as np
# import pandas as pd
# # import torch
# # from .vae_model import VAE

# # 1. App init
# app = FastAPI(title="Churn Prediction Service")

# # 2. Feature schema
# FEATURE_COLUMNS = [
#     "cons_gas_12m",
#     "cons_last_month",
#     "forecast_cons_12m",
#     "forecast_discount_energy",
#     "forecast_meter_rent_12m",
#     "has_gas",
#     "nb_prod_act",
#     "num_years_antig",
#     "pow_max",
#     "days_to_end",
#     "days_since_modif",
#     "days_to_renewal",
#     "last_month_vs_avg",
#     "gas_to_elec_ratio",
#     "margin_ratio",
#     "profitability_index",
#     "is_high_power",
#     "is_low_usage",
#     "latest_price_off_peak_fix",
# ]

# class CustomerFeatures(BaseModel):
#     cons_gas_12m: float
#     cons_last_month: float
#     forecast_cons_12m: float
#     forecast_discount_energy: float
#     forecast_meter_rent_12m: float
#     has_gas: int
#     nb_prod_act: int
#     num_years_antig: int
#     pow_max: float
#     days_to_end: int
#     days_since_modif: int
#     days_to_renewal: int
#     last_month_vs_avg: float
#     gas_to_elec_ratio: float
#     margin_ratio: float
#     profitability_index: float
#     is_high_power: int
#     is_low_usage: int
#     latest_price_off_peak_fix: float

# # 3. Load models on startup
# # device = torch.device("cpu")

# rf_model = None
# logreg_model = None
# svc_model = None
# # vae_model = None
# # vae_scaler = None
# # vae_threshold = None

# @app.on_event("startup")
# def load_models():
#     global rf_model, logreg_model, svc_model

#     # Classic models
#     rf_model = joblib.load("models/rf_balanced.pkl")
#     logreg_model = joblib.load("models/logreg_balanced.pkl")
#     svc_model = joblib.load("models/svc_balanced.pkl")

#     print("Classic ML models loaded successfully.")
#     # VAE removed

# # 4. VAE removed
# # def compute_vae_recon_error(df_row):
# #     return 0.0

# # 5. Health check
# @app.get("/")
# def read_root():
#     return {"status": "ok", "message": "Churn API is running (VAE disabled)."}

# # 6. Predict endpoint
# @app.post("/predict")
# def predict_churn(features: CustomerFeatures):
#     data_dict = features.dict()
#     df_input = pd.DataFrame([data_dict], columns=FEATURE_COLUMNS)

#     # --- Classic models only ---
#     # Random Forest
#     rf_proba = rf_model.predict_proba(df_input)[0, 1]
#     rf_pred = int(rf_proba >= 0.5)

#     # Logistic Regression
#     lr_proba = logreg_model.predict_proba(df_input)[0, 1]
#     lr_pred = int(lr_proba >= 0.5)

#     # SVC
#     svc_proba = svc_model.predict_proba(df_input)[0, 1]
#     svc_pred = int(svc_proba >= 0.5)

#     return {
#         "input": data_dict,
#         "random_forest": {
#             "churn_proba": float(rf_proba),
#             "churn_pred": rf_pred
#         },
#         "logistic_regression": {
#             "churn_proba": float(lr_proba),
#             "churn_pred": lr_pred
#         },
#         "svc": {
#             "churn_proba": float(svc_proba),
#             "churn_pred": svc_pred
#         }
#     }

