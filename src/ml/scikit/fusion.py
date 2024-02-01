#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")
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
    fbeta_score,
)
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .base import run, classification_metrics
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from catboost import CatBoostClassifier


set_config(transform_output="pandas")

RANDOM_SEED = 42

GRID = {
    "svm": {
        "estimator": [
            Pipeline(
                steps=[
                    ("imputer", KNNImputer()),
                    (
                        "model",
                        LinearSVC(
                            random_state=RANDOM_SEED,
                            dual="auto",
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
        ],
        "estimator__model__C": np.logspace(0, -6, num=7),
        "estimator__model__max_iter": [20000],
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


def pipeline_and_grid(key, column_ensemble=None):
    _grid = GRID[key]
    estimator = _grid.pop("estimator")[0]
    steps = []
    selector = ColumnTransformer(
        [("selector", "drop", 0)],
        remainder="passthrough",
    )
    steps.append(("column_selector", selector))
    steps.append(("imputer", KNNImputer()))
    steps.append(("scaler", StandardScaler()))
    grid = {}
    estimators = []
    for i, column_list in enumerate(column_ensemble):
        estimators.append(
            (
                f"estimator_{i}",
                Pipeline(
                    [
                        (
                            "selector",
                            ColumnTransformer(
                                [("selector", "passthrough", column_list)],
                                remainder="drop",
                            ),
                        ),
                        ("model", estimator),
                    ]
                ),
            )
        )
        for key in _grid:
            grid[
                key.replace("estimator", f"stacking_classifier__estimator_{i}__model")
            ] = _grid[key]
    steps.append(
        (
            "stacking_classifier",
            StackingClassifier(
                estimators,
                final_estimator=LinearSVC(dual="auto", class_weight="balanced"),
            ),
        )
    )
    grid["stacking_classifier__final_estimator__C"] = np.logspace(0, -6, num=7)
    grid["stacking_classifier__final_estimator__class_weight"] = [
        "balanced",
        {1: 2, 0: 1},
        {1: 3, 0: 1},
        {1: 4, 0: 1},
    ]
    pipeline = Pipeline(steps)
    return pipeline, grid


if __name__ == "__main__":
    params = dvc.api.params_show()

    label_file = "./data/diagnostics.csv"
    fold_file = "./data/folds.csv"
    feature_folder = f"./data/features/{params['train']['fusion']['features']}"
    result_folder = "./results/fusion"
    metrics_folder = "./metrics/fusion"
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
    scoring = make_scorer(fbeta_score, beta=1)

    run(
        feature_folder,
        label_file,
        fold_file,
        result_folder,
        metrics_folder,
        metadata_file=label_file,
        grid=GRID,
        model_key=params["train"]["fusion"]["estimator"],
        scoring=scoring,
        metrics_fn=classification_metrics,
        group_regex=params["dataset"]["group_regex"],
        target_label=params["train"]["experiment"]["target_label"],
        target_t=params["train"]["experiment"]["target_t"],
        groups=params["dataset"]["groups"],
        leave_out=params["train"]["experiment"]["leave_out"],
        discretisation=None,
        metadata_columns=params["train"]["fusion"]["metadata_columns"],
    )
    with open(os.path.join(result_folder, "params.yaml"), "w") as f:
        yaml.dump(params, f)
