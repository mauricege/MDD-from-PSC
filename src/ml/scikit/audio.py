#!/usr/bin/env python
import os

import dvc.api
import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .base import get_pipeline_and_grid, run
from catboost import CatBoostClassifier

RANDOM_SEED = 42

GRID = {
    "svm": {
        "estimator": [
            LinearSVC(random_state=RANDOM_SEED, dual=True, class_weight="balanced")
        ],
        "estimator__C": np.logspace(0, -6, num=7),
        "estimator__max_iter": [20000],
    },
    "GradientBoostingRegressor": {
        "estimator": [GradientBoostingClassifier(random_state=RANDOM_SEED)],
    },
    "xgb": {
        "estimator": [
            # XGBClassifier(random_state=RANDOM_SEED)
            XGBClassifier(random_state=RANDOM_SEED)
        ],
        # "estimator__learning_rate": [0.3, 0.1, 0.01, 0.001]
    },
    "catboost": {
        "estimator": [
            CatBoostClassifier(
                random_seed=RANDOM_SEED,
                logging_level="Silent",
                auto_class_weights="Balanced",
            )
        ]
    },
}


def classification_metrics(y_true, y_pred, probabilities=None):
    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "acc": float(acc),
        "uar": float(uar),
        "f1": float(f1),
        "prec": float(prec),
        "cm": cm.tolist(),
    }
    return metrics


if __name__ == "__main__":
    params = dvc.api.params_show()

    label_file = "./data/diagnostics.csv"
    fold_file = "./data/folds.csv"
    feature_folder = f"./data/features/{params['train']['audio']['features']}"
    result_folder = "./results/audio"
    metrics_folder = "./metrics/audio"
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
   
    run(
        feature_folder,
        label_file,
        fold_file,
        result_folder,
        metrics_folder,
        grid=GRID,
        # pipeline=pipeline,
        model_key=params["train"]["audio"]["estimator"],
        scoring=make_scorer(accuracy_score),
        metrics_fn=classification_metrics,
        group_regex=params["dataset"]["group_regex"],
        target_label=params["train"]["experiment"]["target_label"],
        target_t=params["train"]["experiment"]["target_t"],
        groups=params["dataset"]["groups"],
        leave_out=params["train"]["experiment"]["leave_out"],
        discretisation=None,
    )
    with open(os.path.join(result_folder, "params.yaml"), "w") as f:
        yaml.dump(params, f)
