import argparse
import os
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI
from churn.data_prep.cleaning import clean_data
from churn.data_prep.loading import file_sha256, load_data, locate_data
from churn.data_prep.splitting import split_data
from churn.features import build_preprocessor
from churn.modeling.evaluate import build_report_artifacts, evaluate_model
from churn.tracking import collect_params, log_run

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Entrena un modelo de churn y registra ejecución en MLflow",
        formatter_class= argparse.ArgumentDefaultsHelpFormatter
    )

    g = p.add_argument_group("modelo")
    g.add_argument("--model", choices=["logreg", "rf"], default="logreg",
                         help="logreg = regresión logística (baseline), rf = Random Forest")
    g.add_argument("--class-weight", choices=["balanced", "none"], default="balanced",
                         help="Manejo del desbalance de clases (aplica a ambos modelos)")
    
    g = p.add_argument_group("Hiperparámetros de regresión logística")
    g.add_argument("--C", type=float, default=1.0, 
                         help="Inverso de la regularización: valores bajos (0.01) simplifican " \
                         "el modelo y altos (10) lo ajustan más a los datos")
    g.add_argument("--penalty", choices=["l1", "l2"], default="l2",
                         help="Tipo de regularización, l1 usa solver liblinear")
    g.add_argument("--max-iter", type=int, default=1000,
                         help="Iteraciones máximas del optimizador")

    g= p.add_argument_group("Hiperparámetros del Randoom Forest")
    g.add_argument("--n-estimators", type=int, default=200, 
                         help="Número de árboles")
    g.add_argument("--max-depth", type=int, default=None, 
                   help="Profundidad máxima de cada árbol")
    g.add_argument("--min-samples-leaf", type=int, default=1,
                   help="Mínimo de muestras por hoja")

    g = p.add_argument_group("Datos y reproducibilidad")
    g.add_argument("--seed", type=int, default=42,
                   help="Semilla aleatoria")
    g.add_argument("--test-size", type=float, default=0.20,
                   help="Porcentaje del conjunto de prueba")
    g.add_argument("--data-path", default=None,
                   help="Ruta a `E Commerce Dataset.xlsx`, si se omite, " \
                   "se descarga en la carpeta data/raw desde kaggle")

    g = p.add_argument_group("MLflow")
    g.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT,
                   help="Experimento donde se agruparán las ejecuciones en MLFlow")
    g.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI), 
                   help="URI de tracking para MLFLow")

    return p.parse_args(argv)

def set_seed(seed:int)->None:
    """Establece la semilla de random y numply"""
    random.seed(seed)
    np.random.seed(seed)

def build_estimator(args):
    """Crea el modelo elegido con los hiperparámetros recibidos"""
    class_weight = None if args.class_weight == "None" else args.class_weight
    if args.model == "logreg":
        solver = "liblinear" if args.penalty == "l1" else "lbfgs"
        return LogisticRegression(C=args.C, penalty=args.penalty, solver=solver,
                                  max_iter=args.max_iter, class_weight=class_weight,
                                  random_state=args.seed)
    return RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                  min_samples_leaf=args.min_samples_leaf, 
                                  class_weight=class_weight, random_state=args.seed,
                                  n_jobs=1)


def train_model(args, X_train, y_train):
    """Entrena el pipelne completo preprocesamiento, clasificador"""
    pipeline = Pipeline([("preprocess", build_preprocessor()),
                         ("model", build_estimator(args))])

    pipeline.fit(X_train, y_train)
    return pipeline

def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    path = locate_data(args.data_path)
    df = clean_data(load_data(path))
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)

    model = train_model(args, X_train, y_train)
    metrics, y_pred, y_prob = evaluate_model(model, X_train, y_train, X_test, y_test)

    params = collect_params(args)
    figures = build_report_artifacts(y_test, y_pred, y_prob)
    run_id, run_name = log_run(args, model, params, metrics, figures, 
                               X_train, X_test, file_sha256(path))

    print(f"""Ejecución {run_name} registrada (run_id={run_id})""")
    print("  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


if __name__ == "__main__":
    main()

    