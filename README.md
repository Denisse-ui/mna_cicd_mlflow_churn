# mna_cicd_mlflow_churn

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Experimentación con MLflow: churn en e-commerce

## Datos

- **Dataset:** *E-Commerce Customer Churn Analysis and Prediction* (Kaggle, `ankitverma2010`).
- **Descarga:** el script lo obtiene con `kagglehub`, fijando la versión 1 del dataset para que
  los resultados no cambien si se actualiza en Kaggle.
- **Copia local:** la primera descarga se guarda en `data/raw/`; las siguientes ejecuciones usan
  esa copia y no requieren conexión. Los datos no se incluyen en el repositorio.
- **Trazabilidad:** cada corrida registra en MLflow la huella SHA-256 del archivo usado.

## Modelos

Se comparan dos modelos de clasificación binaria: 

- Regresión logística (`logreg`), el modelo base (*baseline*) del notebook.
- Random Forest (`rf`)

## Instalación 

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requiere Python 3.14 (probado con 3.14.6).

## Ejecución

Desde la raíz del proyecto, con el entorno virtual activo:

```bash
python -m churn.modeling.train --model logreg --C 1.0 --penalty l2 --class-weight balanced --seed 42
```

Si no se indica `--data-path`, el script usa `data/raw/E Commerce Dataset.xlsx`; si no existe, lo descarga de Kaggle (versión 1) y deja ahí una copia local. Los datos no se versionan.

Cada argumento tiene un valor por defecto. Para ver todos: `python -m churn.modeling.train --help`.

Ejemplo con Random Forest:

```bash
python -m churn.modeling.train --model rf --n-estimators 200 --max-depth 10 --min-samples-leaf 5
```

## Resultados en MLflow

Las corridas se guardan en el experimento `churn-ecommerce-experimentacion` (base SQLite `mlflow.db` y artefactos en `mlartifacts/`, ambos fuera del repositorio). Para verlas:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Abrir http://127.0.0.1:5001. Cada corrida registra:

- **Parámetros:** modelo, hiperparámetros, semilla, tamaño de prueba y manejo de desbalance.
- **Métricas:** accuracy, precision, recall, f1, roc_auc, roc_auc_train y gap_roc_auc.
- **Artefactos:** matriz de confusión y curva ROC.
- **Modelo:** pipeline completo (preprocesamiento + clasificador) con firma y ejemplo de entrada.
- **Etiquetas:** SHA256 del dataset y commit de git (lo registra MLflow).

## Reproducibilidad

- La semilla (`--seed`) se aplica a `random`, `numpy`, la división train/test y el modelo.
- Las dependencias están fijadas en `requirements.txt`.
- El hash SHA256 del dataset queda en cada corrida, para detectar si los datos cambiaron.
- Ejecutar dos veces el mismo comando produce métricas idénticas.

## Estructura del proyecto

```
├── Makefile                 Atajos de ejecución (corridas, UI, pruebas)
├── README.md
├── requirements.txt
├── pyproject.toml
├── churn/
│   ├── config.py            Rutas, constantes y grupos de variables
│   ├── features.py          Preprocesamiento (ColumnTransformer)
│   ├── tracking.py          Registro en MLflow
│   ├── data_prep/
│   │   ├── loading.py       Localización y carga de datos, SHA256
│   │   ├── cleaning.py      Limpieza (duplicados, tipos, categorías, outliers)
│   │   └── splitting.py     División train/test estratificada
│   └── modeling/
│       ├── train.py         Script principal con argumentos de línea de comandos
│       └── evaluate.py      Métricas y gráficas
├── data/raw/                Dataset local (no versionado)
├── docs/                    Documentación del proyecto (mkdocs)
├── notebooks/               EDA y baseline de la semana pasada
├── references/              Decisiones de limpieza
└── tests/
```