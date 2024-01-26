#!/usr/bin/env python
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from range_key_dict import RangeKeyDict
from seglearn.pipe import Pype
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import RocCurveDisplay
from glob import glob

from src.utils import add_groups_to_dataframe


def get_pipeline_and_grid(key, GRID, scorer, aggregate_re=None, column_ensemble=None):
    _grid = GRID[key]
    estimator = _grid.pop("estimator")[0]
    steps = []
    selector = ColumnTransformer(
        [("selector", "drop", 0)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    steps.append(("column_selector", selector))
    steps.append(("scaler", StandardScaler()))
    grid = {}
    estimators = []
    if column_ensemble is not None:
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
                                    verbose_feature_names_out=False,
                                ),
                            ),
                            ("model", estimator),
                        ]
                    ),
                )
            )
            for key in _grid:
                grid[
                    key.replace(
                        "estimator", f"stacking_classifier__estimator_{i}__model"
                    )
                ] = _grid[key]

        steps.append(("stacking_classifier", StackingClassifier(estimators)))
    else:
        steps.append(("estimator", estimator))
        grid = _grid
    pipeline = Pype(steps=steps, scorer=scorer)

    return pipeline, grid


def get_delimiter(file_path, bytes=4096):
    sniffer = csv.Sniffer()
    data = open(file_path).read(bytes)
    try:
        delimiter = sniffer.sniff(data, delimiters=[",", ";", "\t"]).delimiter
    except Exception:
        delimiter = ","
    return delimiter


def make_dict_json_serializable(meta_dict: dict) -> dict:
    cleaned_meta_dict = meta_dict.copy()
    for key in cleaned_meta_dict:
        if type(cleaned_meta_dict[key]) not in [str, float, int]:
            cleaned_meta_dict[key] = str(cleaned_meta_dict[key])
    return cleaned_meta_dict


def parse_labels(
    label_file,
    target_label="HAMD17_ges",
    target_t="matching",
    discretisation=None,
    permutation_state=None,
):
    df = pd.read_csv(label_file)
    df["label"] = df[target_label]
    df = df.dropna(subset=["label"])
    if permutation_state is not None:
        df["label"] = permutation_state.permutation(df["label"].values)
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
    if target_t != "matching":
        df = df[df["t"] == target_t]

    return df[["ID", "label", "t"]], lb


def load_metadata(file, columns):
    df = pd.read_csv(file)
    return df[["ID", "t"] + columns]


def load_features(
    feature_dir,
    label_file,
    fold_file,
    group_regex,
    metadata_file=None,
    metadata_columns=[],
    target_label="HAMD17_ges",
    target_t="matching",
    groups=None,
    leave_out="subjectID",
    discretisation=None,
    permutation_state=None,
    **kwargs,
):
    feature_files = sorted(glob(f"{feature_dir}/*.csv"))

    fold_df = pd.read_csv(fold_file)[["SubjectID", "fold"]].drop_duplicates()
    label_df, lb = parse_labels(
        label_file,
        target_label=target_label,
        target_t=target_t,
        discretisation=discretisation,
        permutation_state=permutation_state,
    )

    fold_names, fold_Xs, fold_ys, fold_groups = [], [], [], []
    for feature_file in feature_files:
        df = pd.read_csv(
            feature_file, delimiter=get_delimiter(feature_file), quotechar="'"
        )

        feature_names = list(df.columns[:]) + metadata_columns

        df = add_groups_to_dataframe(df, group_regex)
        left_on = ["subjectID"]
        right_on = ["ID"]

        if target_t == "matching":
            left_on += ["t"]
            right_on += ["t"]
        merged_df = pd.merge(df, label_df, left_on=left_on, right_on=right_on)
        merged_df = pd.merge(
            merged_df, fold_df, left_on="subjectID", right_on="SubjectID"
        )
        if metadata_file is not None:
            merged_df = pd.merge(
                merged_df,
                load_metadata(metadata_file, metadata_columns),
                left_on=["subjectID", "t"],
                right_on=["ID", "t"],
                how="left",
            )

        if groups:
            for column, values in groups.items():
                if values:
                    merged_df = merged_df[merged_df[column].isin(map(str, values))]

        names = merged_df[df.columns[0]].values.tolist()
        X = merged_df[feature_names]
        if lb:
            y = lb.transform(merged_df["label"])
        else:
            y = merged_df["label"].values.astype(float)
        _groups = merged_df["fold"].tolist()
        fold_names.append(names)
        fold_Xs.append(X)
        fold_ys.append(y)
        fold_groups.append(np.array(_groups))

    return fold_names, fold_Xs, fold_ys, fold_groups, feature_names, lb


def run(
    feature_folder,
    label_file,
    fold_file,
    result_folder,
    metrics_folder,
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
    )
    pipeline, grid = get_pipeline_and_grid(
        model_key,
        grid,
        scoring,
        column_ensemble=[feature_names[1 : -len(metadata_columns)], metadata_columns]
        if len(metadata_columns) > 0
        else None,
    )

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", default=-1)),
        cv=5,
        refit=True,
        verbose=1,
    )

    metrics = []
    prediction_dfs = []

    for fold_index in sorted(set(fold_groups[0])):
        print(f"Computing fold {fold_index}...")
        if len(fold_names) == 1:
            _index = 0
        else:
            _index = fold_index
        X = fold_X[_index]
        y = fold_y[_index]
        groups = fold_groups[_index]
        names = fold_names[_index]

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

        RocCurveDisplay.from_estimator(
            clf.best_estimator_,
            X.iloc[test_indices],
            y_test,
            response_method="predict_proba"
            if pred_proba is not None
            else "decision_function",
            name=type(clf.best_estimator_._final_estimator).__name__,
            plot_chance_level=True,
        )
        plt.show()
        plt.savefig(os.path.join(fold_dir, "roc_curve.pdf"))
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
