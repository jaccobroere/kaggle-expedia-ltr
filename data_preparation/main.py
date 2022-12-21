# %%
from lightgbm import LGBMRanker
import pandas as pd
import os
import numpy as np
import json
from sklearn.model_selection import GroupShuffleSplit
import sweetviz as sv
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from tqdm import tqdm
import numba
from project_modules.ndcg import *


# %%
def load_data(train: bool = True):
    """
    Loads train and test set
    """
    if train:
        s = "training"
    else:
        s = "test"

    path = f"data\{s}_set_VU_DM.csv"
    df = pd.read_csv(path, infer_datetime_format=True, parse_dates=[2])

    return df


def generate_report(df: pd.DataFrame):
    """Generates a EDA report on the dataframe"""
    report = sv.analyze(df)
    report.show_html("SV_report.html")


def add_target(df, target_col="target"):
    """Create target variable 5 if booked, 1 if clicked, 0 for none"""
    df[target_col] = (
        df["click_bool"]
        + (df["booking_bool"] * 5 - df["click_bool"]) * df["booking_bool"]
    )

    return df.copy()


def normalize(df, value_cols, group_col):
    groups = df.groupby(group_col)[value_cols]
    mean, std = groups.transform("mean"), groups.transform("std")
    normalized = (df[mean.columns] - mean) / std

    return normalized.copy()


def featurizing(df: pd.DataFrame):
    # %%
    df = load_data()
    df = add_target(df)
    # %%

    # Create listwise rank features
    features = []

    # Create aggregated features based on prop_id
    agg_dict = dict.fromkeys(df.columns)

    agg = df.groupby("prop_id").agg(agg_dict)

    agg.columns = [
        "_".join(col) if col[-1] != "" else col[0] for col in agg.columns.values
    ]

    # %%

    return df.copy()


def get_balanced_set(df: pd.DataFrame):
    positive = (df["booking_bool"] > 0) | (df["click_bool"] > 0)

    rus = RandomUnderSampler(random_state=2602)

    df_res, _ = rus.fit_resample(df, positive)

    return df_res.copy()


def lightgbm_ranker(df: pd.DataFrame):
    # %%
    # df["target"] = (5 * df["target"]).astype(int)

    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2).split(df, groups=df["srch_id"])
    )

    train, val = df.iloc[train_idx, :], df.iloc[val_idx, :]

    target = "target"
    cols = [
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "prop_starrating",
        "prop_review_score",
        "prop_brand_bool",
        "prop_location_score1",
        "prop_location_score2",
        "prop_log_historical_price",
        "price_usd",
        "srch_length_of_stay",
        "srch_booking_window",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count",
        "srch_query_affinity_score",
        "orig_destination_distance",
    ]

    train_groups = train.groupby("srch_id").size().to_numpy()
    val_groups = val.groupby("srch_id").size().to_numpy()

    params = {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "objective": "lambdarank",
    }

    model = LGBMRanker(**params)
    model.fit(
        train[cols],
        train[target],
        group=train_groups,
        eval_at=[5],
        eval_set=[(val[cols], val[target])],
        eval_group=[val_groups],
    )

    print(
        f"NDCG@5 for validation set: {np.mean(model.evals_result_['valid_0']['ndcg@5'])}"
    )

    # %%
    res = val.reset_index().copy()
    res["rank"] = model.predict(val[cols])
    submission = (
        res[["prop_id", "srch_id", "target", "rank"]]
        .sort_values(by=["srch_id", "rank"], ascending=[True, False])
        .drop("rank", axis=1)
    )

    # %%

    def predict(model, group, features):
        preds = model.predict(group[features])
        ranking = (-preds).argsort()
        ranked_items = group[["prop_id", "srch_id", "target"]].iloc[ranking]

        return ranked_items

    res = pd.DataFrame()

    for srch_id in tqdm(val["srch_id"].unique()):
        sub = val[val["srch_id"] == srch_id]
        res = pd.concat([res, predict(model, sub, cols)], ignore_index=True)

    # %%


@numba.njit()
def predict_fast(model, X: np.ndarray, id_array: np.ndarray, group):

    N = X.shape[0]
    res = np.empty(shape=(N, 3))

    idx = 0
    for i, srch_id in tqdm(enumerate(np.unique(id_array[:, 1]))):
        mask = id_array[:, 1] == srch_id
        preds = model.predict(X[mask])
        ranking = (-preds).argsort()
        ranked_items = id_array[ranking]
        res[idx : (idx + group[i]), :] = ranked_items
        i = idx + group[i]

    return res


# %%
def main():
    # %%
    data = load_data()
    df = add_target(data)
    # %%

    value_cols = [
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "prop_starrating",
        "prop_review_score",
        "prop_brand_bool",
        "prop_location_score1",
        "prop_location_score2",
        "prop_log_historical_price",
        "price_usd",
        "srch_length_of_stay",
        "srch_booking_window",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count",
        "srch_query_affinity_score",
        "orig_destination_distance",
    ]

    aux_cols = [
        "srch_id",
        "date_time",
        "site_id",
        "visitor_location_country_id",
        "prop_id",
        "prop_country_id",
    ]

    # %%
    df = get_balanced_set(data)

    # %% XGBRanker


if __name__ == "__main__":
    # main()
    pass
