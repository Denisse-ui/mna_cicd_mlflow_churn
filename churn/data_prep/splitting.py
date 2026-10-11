import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import TARGET

def split_data(df: pd.DataFrame, test_size: float, seed: int):
    """División estratificada por Churn, reproducible a través de semillas"""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)