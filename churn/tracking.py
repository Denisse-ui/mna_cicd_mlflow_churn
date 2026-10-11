import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from churn.config import ARTIFACT_ROOT

def collect_params(args)-> dict:
    """Obtiene los parámetros del modelo elegido, además de los de la ejecución"""
    params = {"model": args.model, "class_weight":args.class_weight,
              "seed": args.seed, "test_size": args.test_size}

    if args.model == "logreg":
        params.update({"C": args.C, "penalty": args.penalty, "max_iter": args.max_iter})
    else:
        params.update({"n_estimators": args.n_estimators,
                       "max_depth": args.max_depth,
                       "min_samples_leaf": args.min_samples_leaf
                       })
    return params


def setup_experiment(tracking_uir:str, experiment_name: str)-> None:
    """Asigna el servidor y crea el experimento con artefactos dentro del proyecto"""
    mlflow.set_tracking_uri(tracking_uir)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_ROOT.as_uri())

    mlflow.set_experiment(experiment_name)

def build_run_name(args) ->str:
    """Nombre de la ejecución a partir de sus hiperparámetros"""
    if args.model == "logreg":
        hiper = f"""C{args.C}-{args.penalty}"""
    else:
        depth = args.max_depth if args.max_depth is not None else "none"
        hiper = f"""n{args.n_estimators}-d{depth}-leaf{args.min_samples_leaf}"""

    return f"""{args.model}-{hiper}-{args.class_weight}-s{args.seed}"""


def log_run(args, model, params, metrics, figures, X_train, X_test, data_hash):
    "Regustra los pamatros, métricas, gráficos y modelo de la ejecución"
    setup_experiment(args.tracking_uri, args.experiment_name)


    with mlflow.start_run(run_name=build_run_name(args)) as run:
        mlflow.set_tags({"dataset": "E Commerce Customer Churn Kaggle",
                         "dataset_sha256": data_hash})

        mlflow.log_params(params)
        mlflow.log_params({"n_train": len(X_train),
                          "n_test": len(X_test)})


        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        for filename, fig in figures.items():
            mlflow.log_figure(fig, filename)

            plt.close(fig)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model = model, name= "modelo", signature=signature, 
            input_example=X_train.head(),
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"]
        )

        return run.info.run_id, run.info.run_name
