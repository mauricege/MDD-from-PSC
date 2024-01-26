import subprocess as sp
import tempfile
from os.path import join
from os import makedirs

import pandas as pd
import regex as re
from seglearn.transform import XyTransformerMixin
from sklearn.base import BaseEstimator
from tqdm import tqdm
from cacheout import Cache
from src.utils import add_groups_to_dataframe
import dvc.api

cache = Cache(maxsize=4)

GRID = {
    "codebook_size": [2000],
    "n_assigned": [50],
}


class XBOWTransformer(BaseEstimator, XyTransformerMixin):
    def __init__(
        self,
        codebook_size=2000,
        n_assigned=50,
        aggregate_re=None,
        aggregate_column=None,
        openxbow_jar="openXBOW/openXBOW.jar",
        heap_size="32g",
        label_aggregation="first",
        random_state=42,
        **kwargs,
    ):
        self.codebook_size = codebook_size
        self.n_assigned = n_assigned
        self.openxbow_jar = openxbow_jar
        assert aggregate_re, "An aggregate regex has to be given!"
        self.aggregate_re = aggregate_re

        self.aggregate_column = aggregate_column
        self.label_aggregation = label_aggregation
        self.codebook = None
        self.random_state = random_state

        self.heap_size = heap_size
        self.openxbow_jar = openxbow_jar

        self._cached_aggregate_column = None

    @staticmethod
    @cache.memoize()
    def __run_xbow(
        X,
        y,
        codebook=None,
        codebook_size=2000,
        n_assigned=50,
        heap_size="32g",
        openxbow_jar="openXBOW/openXBOW.jar",
        random_state=42,
        aggregate_column=None,
        aggregate_re=None,
        label_aggregation="first",
    ):
        xbow_base_cmd = [
            "java",
            f"-Xmx{heap_size}",
            "-jar",
            openxbow_jar,
        ]
        xbow_options = [
            "-writeName",
            "-csvHeader",
            "-size",
            str(codebook_size),
            "-a",
            str(n_assigned),
            "-standardizeInput",
            "-log",
            "-seed",
            str(random_state),
        ]
        X = X.copy().reset_index(drop=True)

        if aggregate_column is None:
            aggregate_column = X.columns[0]
        assert aggregate_column in list(
            X.columns
        ), f"Aggregation column {aggregate_column} is missing from input columns: {list(X.columns)}."

        aggregate_fn = lambda x: re.match(aggregate_re, x).group(1)
        X.loc[:, aggregate_column] = X[aggregate_column].apply(func=aggregate_fn)
        if y is not None:
            _y_agg = pd.DataFrame()

            if len(y.shape) == 1:
                _y_agg["label"] = y
                label_columns = ["label"]
            else:
                label_columns = []
                for i in range(y.shape[1]):
                    _y_agg[f"label{i}"] = y[:, i]
                    label_columns.append(f"label{i}")

            _y_agg.loc[:, aggregate_column] = X.loc[:, aggregate_column]
            _y_agg = (
                _y_agg.groupby(aggregate_column).agg(label_aggregation).reset_index()
            )
        with tempfile.TemporaryDirectory() as tmpdirname:
            # print("created temporary directory", tmpdirname)
            input_features = join(tmpdirname, "input_features.csv")
            temp_features = join(tmpdirname, "features.csv")
            codebook_path = join(tmpdirname, "codebook")
            X.to_csv(input_features, sep=";", index=False, header=False)
            xbow_cmd = [] + xbow_base_cmd
            xbow_cmd += [
                "-i",
                input_features,
                "-o",
                temp_features,
            ]
            if codebook is None:
                xbow_cmd += [
                    "-B",
                    codebook_path,
                ]
                xbow_cmd += xbow_options
            else:
                with open(codebook_path, "w") as f:
                    f.write(codebook)
                xbow_cmd += [
                    "-writeName",
                    "-csvHeader",
                    "-b",
                    codebook_path,
                ]
            assert sp.run(
                xbow_cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL
            ), "Failed to run openXBoW!"
            with open(codebook_path) as f:
                codebook = f.read()

            X = pd.read_csv(temp_features, delimiter=";", quotechar="'")
        X.rename(columns={X.columns[0]: aggregate_column}, inplace=True)
        _cached_aggregate_column = X[aggregate_column].copy()
        if y is not None:
            X = X.merge(_y_agg, on=aggregate_column)
            y = X.pop(*label_columns).values
        return X, y, codebook, aggregate_column, _cached_aggregate_column

    def fit(self, X, y, sample_weight=None):
        (
            _,
            _,
            self.codebook,
            self.aggregate_column,
            self._cached_aggregate_column,
        ) = XBOWTransformer.__run_xbow(
            X,
            y,
            codebook=self.codebook,
            codebook_size=self.codebook_size,
            n_assigned=self.n_assigned,
            heap_size=self.heap_size,
            openxbow_jar=self.openxbow_jar,
            random_state=self.random_state,
            aggregate_column=self.aggregate_column,
            aggregate_re=self.aggregate_re,
            label_aggregation=self.label_aggregation,
        )
        return self

    def transform(self, X, y, sample_weight=None):
        assert self.codebook is not None, "Not fitted yet!"
        (
            X,
            y,
            self.codebook,
            self.aggregate_column,
            self._cached_aggregate_column,
        ) = XBOWTransformer.__run_xbow(
            X,
            y,
            codebook=self.codebook,
            codebook_size=self.codebook_size,
            n_assigned=self.n_assigned,
            heap_size=self.heap_size,
            openxbow_jar=self.openxbow_jar,
            random_state=self.random_state,
            aggregate_column=self.aggregate_column,
            aggregate_re=self.aggregate_re,
            label_aggregation=self.label_aggregation,
        )
        return X, y, None

    def fit_transform(self, X, y, sample_weight=None):
        (
            X,
            y,
            self.codebook,
            self.aggregate_column,
            self._cached_aggregate_column,
        ) = XBOWTransformer.__run_xbow(
            X,
            y,
            codebook=self.codebook,
            codebook_size=self.codebook_size,
            n_assigned=self.n_assigned,
            heap_size=self.heap_size,
            openxbow_jar=self.openxbow_jar,
            random_state=self.random_state,
            aggregate_column=self.aggregate_column,
            aggregate_re=self.aggregate_re,
            label_aggregation=self.label_aggregation,
        )
        return X, y, None


if __name__ == "__main__":
    params = dvc.api.params_show()

    X = pd.read_csv(
        f'data/features/{params["xbow"]["features"]}/features.csv',
        delimiter=",",
        quotechar="'",
    )
    feature_columns = list(X.columns)

    X = add_groups_to_dataframe(X, params["dataset"]["group_regex"])
    if params["dataset"]["groups"]:
        for column, values in params["dataset"]["groups"].items():
            if values:
                X = X[X[column].isin(map(str, values))]
    folds = pd.read_csv("./data/folds.csv")[
        ["SubjectID", "fold", "t"]
    ].drop_duplicates()

    merged_df = pd.merge(
        X, folds, left_on=["subjectID", "t"], right_on=["SubjectID", "t"]
    )

    xbow = XBOWTransformer(
        aggregate_re=params["dataset"]["aggregate_re"], heap_size="32G"
    )
    makedirs("data/features/xbow", exist_ok=True)

    for fold in tqdm(sorted(set(merged_df["fold"].values))):
        fold_train = merged_df[merged_df["fold"] != fold][feature_columns]
        fold_test = merged_df[merged_df["fold"] == fold][feature_columns]
        fold_train, _, _ = xbow.fit_transform(fold_train, None)
        fold_test, _, _ = xbow.transform(fold_test, None)
        fold_train[fold_train.columns[0]] += ".wav"
        fold_test[fold_test.columns[0]] += ".wav"
        fold_df = pd.concat([fold_train, fold_test])
        fold_df.to_csv(f"./data/features/xbow/fold_{fold}.csv", index=False)
