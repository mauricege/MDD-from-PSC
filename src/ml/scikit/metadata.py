#!/usr/bin/env python
import os

import dvc.api
import numpy as np
import yaml
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import json

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

from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from range_key_dict import RangeKeyDict
from catboost import CatBoostClassifier
from .base import get_pipeline_and_grid, make_dict_json_serializable

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
                            random_state=RANDOM_SEED, dual=True, class_weight="balanced"
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


def load(
    label_file, target_label="HAMD17_ges", discretisation=None, metadata_columns=[]
):
    df = pd.read_csv(label_file)
    df["label"] = df[target_label]
    df = df.dropna(subset=["label"])
    lb = None
    if discretisation:
        discretisation_steps = sorted(map(int, discretisation.keys())) + [sys.maxsize]
        mapping = RangeKeyDict(
            {
                (discretisation_steps[i], discretisation_steps[i + 1]): discretisation[
                    discretisation_steps[i]
                ]
                for i in range(len(discretisation))
            }
        )
        df["label"] = df["label"].apply(lambda label: mapping[label])
        lb = LabelEncoder().fit(
            [discretisation[key] for key in discretisation_steps[:-1]]
        )

    return (
        df[["ID", "label", "t", "fold"] + metadata_columns].drop_duplicates(
            ignore_index=True
        ),
        lb,
    )


def run(
    fold_file,
    result_folder,
    metrics_folder,
    grid,
    scoring,
    metrics_fn,
    target_label="HAMD17_ges",
    model_key="svm",
    groups=None,
    metadata_file=None,
    metadata_columns=[],
    cv=5,
    discretisation=None,
):
    df, lb = load(
        fold_file,
        target_label=target_label,
        discretisation=discretisation,
        metadata_columns=metadata_columns,
    )
    X = df[["ID"] + metadata_columns]
    y = df["label"]
    names = df["ID"] + "/" + df["t"]
    groups = df["fold"]
    pipeline, grid = get_pipeline_and_grid(
        model_key,
        grid,
        scoring,
        column_ensemble=None,
    )

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", default=-1)),
        cv=cv,
        refit=True,
        verbose=1,
    )

    metrics = []
    prediction_dfs = []

    for fold_index in sorted(set(groups)):
        print(f"Computing fold {fold_index}...")

        train_indices = np.argwhere(groups != fold_index).squeeze()
        test_indices = np.argwhere(groups == fold_index).squeeze()

        fold_dir = os.path.join(result_folder, str(fold_index))
        os.makedirs(fold_dir, exist_ok=True)

        clf.fit(X.iloc[train_indices], y[train_indices])

        preds = clf.best_estimator_.predict(X.iloc[test_indices])
        y_test = y[test_indices]

        if hasattr(clf.best_estimator_._final_estimator, "predict_proba"):
            pred_proba = clf.predict_proba(X.iloc[test_indices])
        else:
            pred_proba = None

        test_names = np.array(names)[test_indices]

        if lb:
            preds, y_test = lb.inverse_transform(preds), lb.inverse_transform(y_test)
        prediction_df = pd.DataFrame(
            data={"filename": test_names, "prediction": preds, "true": y_test}
        )
        if pred_proba is not None:
            class_names = lb.classes_ if lb else list(range(pred_proba.shape[1]))

            prediction_df[
                [f"proba_{class_name}" for class_name in class_names]
            ] = pred_proba

        svm_params = make_dict_json_serializable(clf.best_params_)
        prediction_dfs.append(prediction_df)
        fold_metrics = metrics_fn(y_test, preds, probabilities=pred_proba)
        metrics.append(fold_metrics)

        prediction_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
        with open(os.path.join(fold_dir, "best_params.json"), "w") as f:
            json.dump(svm_params, f)

        pd.DataFrame(clf.cv_results_).to_csv(
            os.path.join(fold_dir, "grid_search.csv"), index=False
        )
        with open(os.path.join(fold_dir, "metrics.yaml"), "w") as f:
            yaml.dump(fold_metrics, f)

    df_predictions = pd.concat(prediction_dfs)
    df_predictions.to_csv(os.path.join(result_folder, "predictions.csv"), index=False)
    metrics = {
        metric_key: {
            "fold_values": [fold_metrics[metric_key] for fold_metrics in metrics],
            "mean": float(
                np.mean([fold_metrics[metric_key] for fold_metrics in metrics])
            ),
            "stdv": float(
                np.std([fold_metrics[metric_key] for fold_metrics in metrics])
            ),
        }
        for metric_key in metrics[0]
    }

    with open(os.path.join(metrics_folder, "metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)
    with open(os.path.join(result_folder, "metrics.yaml"), "w") as f:
        yaml.dump(metrics, f)

    return metrics


if __name__ == "__main__":
    params = dvc.api.params_show()

    fold_file = "./data/folds.csv"
    result_folder = "./results/metadata"
    metrics_folder = "./metrics/metadata"
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)

    run(
        fold_file,
        result_folder,
        metrics_folder,
        grid=GRID,
        model_key=params["train"]["metadata"]["estimator"],
        scoring=make_scorer(accuracy_score),
        metrics_fn=classification_metrics,
        target_label=params["train"]["experiment"]["target_label"],
        discretisation=None,
        metadata_columns=params["train"]["metadata"]["metadata_columns"],
    )
    with open(os.path.join(result_folder, "params.yaml"), "w") as f:
        yaml.dump(params, f)
