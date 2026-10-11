import json
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg") ##Permite ejecutar el script en terminal
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                            f1_score, precision_score, recall_score, roc_auc_score)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Métricas definidas par el modelo, más curva ROC-AUC para revisar sobreajuste"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "roc_auc_train": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
     }
    metrics["gap_roc_auc"] = metrics["roc_auc_train"] - metrics["roc_auc"]

    return metrics, y_pred, y_prob

def build_report_artifacts(y_test, y_pred, y_prob): 
    """Genera la matriz de confusión y la curva ROC"""
    fig_cm, ax = plt.subplots(figsize = (5,4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Permanece", "Abandona"], cmap="Blues", ax=ax
    )
    ax.set_title("Matriz de confusión (prueba)")
    ax.grid(False)

    fig_roc, ax = plt.subplots(figsize= (5,4))
    RocCurveDisplay.from_predictions(y_test, y_prob, name="Modelo", ax=ax)
    ax.plot([0,1], [0,1], "k--", label="Azar")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")

    return {"matriz_confusion.png": fig_cm, "curva_roc.png": fig_roc}