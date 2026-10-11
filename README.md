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

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         churn and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── churn   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes churn a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

