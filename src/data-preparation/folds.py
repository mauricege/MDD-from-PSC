import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import dvc.api


if __name__ == "__main__":
    params = dvc.api.params_show()
    df = pd.read_csv("./data/labels.csv")
    diagnostics = pd.read_csv("./data/diagnostics.csv")
    df = pd.merge(df, diagnostics, left_on=["SubjectID", "t"], right_on=["ID", "t"])
    if params["folds"]["other_scales"]:
        df = df.dropna().reset_index().drop(columns=["index"])

    fold_dfs = []
    for i, (_, test_indices) in enumerate(
        StratifiedGroupKFold(
            n_splits=params["folds"]["cv"], shuffle=True, random_state=42
        ).split(X=df, y=df[params["folds"]["stratify"]], groups=df["SubjectID"])
    ):
        fold_df = df.loc[test_indices].copy()
        fold_df["fold"] = i
        fold_dfs.append(fold_df)

    folds = pd.concat(fold_dfs)
    folds.to_csv("./data/folds.csv", index=False)
