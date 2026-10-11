import pandas as pd
import numpy as np

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """3.1 Elimina duplicados del df sin considerar CustomerID"""
    df = df.copy()
    cols_sin_id = [col for col in df.columns if col != "CustomerID"]
    return df.drop_duplicates(subset=cols_sin_id, keep="first").reset_index(drop=True)


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """3.2 Corrige CustomerID como indice y CityTier como categoría """
    df = df.copy().set_index("CustomerID")
    df["CityTier"] = df["CityTier"].astype("category")
    return df

def unify_categories(df: pd.DataFrame) -> pd.DataFrame:
    """3.3 Categorias diferentes que representan lo mismo"""
    df = df.copy()
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace({
        "Phone": "Mobile Phone"
    })
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({
        "CC": "Credit Card", "COD": "Cash on Delivery"
    })
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace({
        "Mobile": "Mobile Phone"
    })
    return df

def mask_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """3.4: WarehouseToHome por encima del límite IQR pasa a faltante."""
    df = df.copy()
    serie = df["WarehouseToHome"].dropna()
    q1, q3 = serie.quantile([0.25, 0.75])
    limite = q3 + 1.5 * (q3 - q1)
    df.loc[df["WarehouseToHome"] > limite, "WarehouseToHome"] = np.nan
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las decisiones de limpieza, en orden, sin modificar el original."""
    df = remove_duplicates(df)
    df = fix_types(df)
    df = unify_categories(df)
    return mask_outliers(df)