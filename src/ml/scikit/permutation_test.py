#!/usr/bin/env python
import os
import pandas as pd

import dvc.api
import numpy as np
import yaml
from copy import deepcopy
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
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from .base import get_pipeline_and_grid, load_features
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
        "estimator": [XGBClassifier(random_state=RANDOM_SEED)],
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


def run(
    feature_folder,
    label_file,
    fold_file,
    grid,
    scoring,
    metrics_fn,
    group_regex,
    target_label="HAMD17_ges",
    model_key="svm",
    target_t="matching",
    groups=None,
    leave_out="subjectID",
    metadata_file=None,
    metadata_columns=[],
    cv=5,
    discretisation=None,
    permutation_state=None,
):
    fold_names, fold_X, fold_y, fold_groups, feature_names, lb = load_features(
        feature_folder,
        label_file,
        fold_file,
        group_regex=group_regex,
        target_label=target_label,
        target_t=target_t,
        groups=groups,
        leave_out=leave_out,
        metadata_file=metadata_file,
        metadata_columns=metadata_columns,
        discretisation=discretisation,
        permutation_state=permutation_state,
    )
    pipeline, _grid = get_pipeline_and_grid(
        model_key,
        grid,
        scoring,
        column_ensemble=[feature_names[1 : -len(metadata_columns)], metadata_columns]
        if len(metadata_columns) > 0
        else None,
    )

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=_grid,
        n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", default=-1)),
        cv=5,
        refit=True,
        verbose=1,
    )

    metrics = []

    # compute non-permuted score
    for fold_index in sorted(set(fold_groups[0])):
        print(f"Computing fold {fold_index}...")
        if len(fold_names) == 1:
            _index = 0
        else:
            _index = fold_index
        X = fold_X[_index]
        y = fold_y[_index]
        groups = fold_groups[_index]

        train_indices = np.argwhere(groups != fold_index).squeeze()
        test_indices = np.argwhere(groups == fold_index).squeeze()

        clf.fit(X.iloc[train_indices], y[train_indices])

        preds = clf.best_estimator_.predict(X.iloc[test_indices])
        y_test = y[test_indices]

        if hasattr(clf.best_estimator_._final_estimator, "predict_proba"):
            pred_proba = clf.predict_proba(X.iloc[test_indices])
        else:
            pred_proba = None

        fold_metrics = metrics_fn(y_test, preds, probabilities=pred_proba)
        metrics.append(fold_metrics)

    score = {
        metric_key: np.mean([fold_metrics[metric_key] for fold_metrics in metrics])
        for metric_key in metrics[0]
    }
    return score


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
    score = run(
        feature_folder,
        label_file,
        fold_file,
        grid=deepcopy(GRID),
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
    random_state = np.random.RandomState(42)
    permutation_scores = []
    for i in tqdm(range(100)):
        permutation_scores.append(
            run(
                feature_folder,
                label_file,
                fold_file,
                grid=deepcopy(GRID),
                model_key=params["train"]["audio"]["estimator"],
                scoring=make_scorer(accuracy_score),
                metrics_fn=classification_metrics,
                group_regex=params["dataset"]["group_regex"],
                target_label=params["train"]["experiment"]["target_label"],
                target_t=params["train"]["experiment"]["target_t"],
                groups=params["dataset"]["groups"],
                leave_out=params["train"]["experiment"]["leave_out"],
                discretisation=None,
                permutation_state=random_state,
            )
        )
    df = pd.DataFrame(data=[score] + permutation_scores)
    df.to_csv("permutation_scores.csv", index=False)
