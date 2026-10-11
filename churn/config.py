"""Constantes del proyecto: rutas, nombres y grupos de variables."""
from pathlib import Path

# Rutas
PROJ_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJ_ROOT / "data" / "raw"        # data/raw dentro del proyecto

# Datos
KAGGLE_DATASET = "ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/versions/1"    # dueño/nombre/versions/1
DATA_FILE = "E commerce Dataset.xlsx"         # nombre del archivo Excel
DATA_SHEET = "E comm"        # nombre de la hoja con los datos
TARGET = "Churn"            # variable objetivo

# MLflow
DEFAULT_EXPERIMENT = "churn-ecommerce-experimentacion"
DEFAULT_TRACKING_URI = f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"      # sqlite:/// + ruta a mlflow.db en la raíz
ARTIFACT_ROOT = PROJ_ROOT / "mlartifacts"             # carpeta mlartifacts en la raíz

# Grupos de variables
NUM_CON_NULOS = ["CouponUsed", "OrderCount", "OrderAmountHikeFromlastYear", "Tenure",
                 "WarehouseToHome", "HourSpendOnApp", "DaySinceLastOrder"]
NUM_SIN_NULOS = ["NumberOfDeviceRegistered", "NumberOfAddress", "CashbackAmount",
                 "SatisfactionScore"]
COLS_BINARIAS = ["Complain"]
COLS_CATEGORICAS = ["PreferredLoginDevice", "PreferredPaymentMode", "Gender",
                    "PreferedOrderCat", "MaritalStatus", "CityTier"]