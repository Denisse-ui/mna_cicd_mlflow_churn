from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import (COLS_BINARIAS, COLS_CATEGORICAS, NUM_CON_NULOS, NUM_SIN_NULOS)

def build_preprocessor() -> ColumnTransformer:
    """Mediana + indicadores de faltante y escalado; one-hot en categóricas"""
    num_nulos = Pipeline([
        ("Imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("Scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("num_con_nulos", num_nulos, NUM_CON_NULOS), 
        ("num_sin_nulos", StandardScaler(), NUM_SIN_NULOS), 
        ("binarias", "passthrough", COLS_BINARIAS),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), COLS_CATEGORICAS),
    ])