import hashlib
import shutil
from pathlib import Path

import pandas as pd
from churn.config import DATA_FILE, DATA_SHEET, KAGGLE_DATASET, RAW_DATA_DIR


def load_data(path:str)-> pd.DataFrame:
    """Carga el archivo excel en un DataFrame de pandas"""
    return pd.read_excel(path, sheet_name=DATA_SHEET)

def file_sha256(path:str)-> str:
    """Genera un has que asegura la versión de cada archivo"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter (lambda: f.read(1<<20), b""):
            h.update(chunk)
        return h.hexdigest()

def locate_data (data_path:str | Path | None= None)-> Path:
    """Si data_path != None devuelve la ruta, 
        en otro caso crea una copua en local"""
    import kagglehub 

    if data_path:
        return data_path

    local = RAW_DATA_DIR/DATA_FILE
    if local.exists():
        return local

    download = Path(kagglehub.dataset_download(KAGGLE_DATASET)) / DATA_FILE
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(download, local)
    return local
    

    


