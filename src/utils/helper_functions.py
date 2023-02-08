# %%

import datetime as dt
import os
import pickle

import numpy as np
import optuna
import pandas as pd
import sweetviz as sv
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


# %%
def load_raw_data(train: bool = True):
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
    """
    Generates a EDA report on the dataframe
    """
    report = sv.analyze(df)
    report.show_html("SV_report.html")


def add_target(df, target_col="target"):
    """
    Create target variable 5 if booked, 1 if clicked, 0 for none
    """
    df[target_col] = (
        df["click_bool"]
        + (df["booking_bool"] * 5 - df["click_bool"]) * df["booking_bool"]
    )

    return df


def norm_and_fill_null(x):
    mean, std = x.mean(), x.std()
    if std == 0:
        return 0
    else:
        return (x - mean) / std


def add_norm_features_for_value(df, value_col="price_usd"):

    df["norm_" + value_col + "_wrt_srch_id"] = df.groupby("srch_id")[
        value_col
    ].transform(norm_and_fill_null)

    df["norm_" + value_col + "_wrt_prop_id"] = df.groupby("prop_id")[
        value_col
    ].transform(norm_and_fill_null)

    df["norm_" + value_col + "_wrt_srch_destination_id"] = df.groupby(
        "srch_destination_id"
    )[value_col].transform(norm_and_fill_null)

    df["month"] = pd.to_datetime(df["date_time"]).dt.month
    df["norm_" + value_col + "_wrt_month"] = df.groupby("month")[value_col].transform(
        norm_and_fill_null
    )

    df["norm_" + value_col + "_wrt_srch_booking_window"] = df.groupby(
        "srch_booking_window"
    )[value_col].transform(norm_and_fill_null)

    df["norm_" + value_col + "_wrt_prop_country_id"] = df.groupby("prop_country_id")[
        value_col
    ].transform(norm_and_fill_null)

    return df


def LinModel(df):
    y = df[["prop_starrating"]].values
    X = df[["price_usd"]].values
    return LinearRegression().fit(X, y)


def ModelperCountry(df):
    series = (
        df.iloc[np.where(df["prop_starrating"] != 0)]
        .groupby("prop_country_id")
        .apply(LinModel)
    )
    return series


def ImputeStarrating0(df, series):

    for i in np.where(df["prop_starrating"] == 0)[0]:
        if df.iloc[i]["prop_country_id"] in series.index:
            df.at[i, "prop_starrating"] = series[df.iloc[i]["prop_country_id"]].predict(
                [[df.iloc[i]["price_usd"]]]
            )
        else:
            df.at[i, "prop_starrating"] = 0

    df["prop_starrating"] = df["prop_starrating"].apply(
        lambda x: 5 if x > 5 else (0 if x < 0 else x)
    )

    return df


def get_balanced_set(df: pd.DataFrame):
    positive = (df["booking_bool"] > 0) | (df["click_bool"] > 0)

    rus = RandomUnderSampler(random_state=2602)

    df_res, _ = rus.fit_resample(df, positive)

    return df_res


def normalize(df, value_cols, group_col):
    groups = df.groupby(group_col)[value_cols]
    mean, std = groups.transform("mean"), groups.transform("std")
    normalized = (df[mean.columns] - mean) / std

    return normalized


def null_impute_value(df, columns_to_impute, value=0):
    """
    Imputes null values in given columns with 0 and returns DataFrame
    :param df: pd.DataFrame
    :param columns_to_impute: array-like
    returns pd.DataFrame
    """
    df.loc[:, columns_to_impute] = df.loc[:, columns_to_impute].fillna(value)

    return df


def impute_over_group(df, group_cols, columns_to_impute, func=None):
    if not func:

        def func(x):
            return x.fillna(x.median())

    df.loc[:, columns_to_impute] = df.groupby(group_cols)[columns_to_impute].transform(
        lambda x: x.fillna(x.median())
    )

    return df


def mean_minmax(x):
    """
    returns mean of minmaxed normalized x
    :param x: array-like
    """
    maxx, minn = x.max(), x.min()

    if maxx == minn:
        return 0
    else:
        return ((x - minn) / (maxx - minn)).mean()


def target_encoding_values(df, group_cols, target_cols, agg_func=mean_minmax):
    columns = ["_".join([s, "encoded"]) for s in target_cols]
    grouper = df.groupby(group_cols)[target_cols].agg(agg_func)
    grouper.columns = columns

    return grouper


def enrich_target_encoding(to_enrich, encoded_values, group_col):
    res = pd.merge(
        to_enrich,
        encoded_values,
        left_on=group_col,
        right_index=True,
        how="left",
        suffixes=("", "_encoded"),
    )

    return res.copy()


def date_str():
    return dt.datetime.now().strftime("%Y%m%d_%H%M")


def train_val_split_group(df, test_size=0.1, group_key="srch_id"):
    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size).split(
            df, groups=df[group_key]
        )
    )

    train, val = df.iloc[train_idx, :], df.iloc[val_idx, :]

    return train.copy(), val.copy()


def main():
    pass


def parse_best_params_from_csv(path) -> dict:
    df = pd.read_csv(path)
    params = {
        "_".join(col.split("_")[1:]): df.loc[df.value.argmax(), col]
        for col in df.columns
        if "params" in col
    }

    return params


if __name__ == "__main__":
    wd = r"C:\Users\jacbr\OneDrive\Documenten\vu-data-mining-techniques\Assignment 2"
    os.chdir(wd)

    main()

# %% ARCHIVE

# def normalizer(df: pd.DataFrame):
#     """Normalizes monetary values to make them more comparable across key indicators"""
#     value_cols = [
#         "visitor_hist_starrating",
#         "visitor_hist_adr_usd",
#         "prop_starrating",
#         "prop_review_score",
#         "prop_brand_bool",
#         "prop_location_score1",
#         "prop_location_score2",
#         "prop_log_historical_price",
#         "price_usd",
#         "srch_length_of_stay",
#         "srch_booking_window",
#         "srch_adults_count",
#         "srch_children_count",
#         "srch_room_count",
#         "srch_query_affinity_score",
#         "orig_destination_distance",
#     ]

#     aux_cols = [
#         "srch_id",
#         "date_time",
#         "site_id",
#         "visitor_location_country_id",
#         "prop_id",
#         "prop_country_id",
#     ]

#     # Transform features based on prop_country_id
#     groups = df.groupby("prop_country_id")[value_cols]
#     mean, std = groups.transform("mean"), groups.transform("std")
#     normalized = (df[mean.columns] - mean) / std

#     return normalized


# def featurizing(df: pd.DataFrame):
#     # Create listwise rank features
#     features = []

#     # Create aggregated features based on prop_id
#     agg_dict = dict.fromkeys(df.columns)

#     agg = df.groupby("prop_id").agg(agg_dict)

#     agg.columns = [
#         "_".join(col) if col[-1] != "" else col[0] for col in agg.columns.values
#     ]

#     return df.copy()

# def full_imputation_pipe(df):
#     df = null_impute_value(
#         df,
#         [
#             "prop_review_score",
#             "prop_location_score2",
#             "srch_query_affinity_score",
#             "booking_bool_encoded",
#             "click_bool_encoded",
#             "target_encoded",
#         ],
#         value=0,
#     )

#     df = null_impute_value(
#         df,
#         [
#             "position_encoded",
#         ],
#         value=1,
#     )

#     df = impute_over_group(
#         df,
#         ["visitor_location_country_id", "prop_country_id"],
#         "orig_destination_distance",
#     )

#     df = impute_over_group(df, ["prop_country_id"], "orig_destination_distance")
#     df["orig_destination_distance"] = df["orig_destination_distance"].transform(
#         lambda x: x.fillna(x.median())
#     )

#     return df.copy()

# def impute_enrich(df, enrich_dict):

#     for key in enrich_dict:
#         df = enrich_target_encoding(df, enrich_dict[key], key)

#     df = full_imputation_pipe(df)

#     return df.copy()


def predict_per_group(
    model, X: np.ndarray, id_array: np.ndarray, group: np.ndarray, columns: list
):
    """
    Predicts based on predictions per group, needs srch_id in the first column
    """

    N = X.shape[0]
    K = id_array.shape[1]
    res = np.empty(shape=(N, K), dtype=int)

    idx = 0
    for i, srch_id in enumerate(np.unique(id_array[:, 0])):
        mask = id_array[:, 0] == srch_id
        preds = model.predict(X[mask])

        ranking = (-preds).argsort() + idx
        ranked_items = id_array[ranking]

        res[np.min(ranking) : (np.max(ranking) + 1), :] = ranked_items

        idx += group[i]

    res = pd.DataFrame(res, columns=columns, dtype=int)

    return res


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def predict_in_batches(model, df, cols, id_cols, batch_size=20000):
    n_batches = int(df.shape[0] / batch_size)
    res = pd.DataFrame()

    ids = list(df["srch_id"].unique())

    ch = chunks(ids, int(len(ids) / n_batches))

    for srch_ids in tqdm(ch):
        temp = df.loc[df["srch_id"].isin(srch_ids), :].copy()

        X, id_array = temp[cols].to_numpy(), temp[id_cols].to_numpy()
        group = temp.groupby("srch_id").size().to_numpy()
        preds = predict_per_group(
            model, X, id_array, group, columns=["srch_id", "prop_id"]
        )

        res = pd.concat([res, preds])

    return res


def ndcg_at_k(target, k=5):
    k = min([k, target.shape[0]])
    idx = np.log2(np.array([i + 2 for i in range(k)]))
    dcg = (target[:k] / idx).sum()

    idcg = (np.sort(target)[::-1][:k] / idx).sum()

    return dcg / idcg


def calc_ndcg_submission(submission, df, k=5):
    ranking = pd.merge(
        left=submission,
        right=df,
        left_on=["srch_id", "prop_id"],
        right_on=["srch_id", "prop_id"],
    )[["srch_id", "prop_id", "target"]]

    res = ranking.groupby("srch_id")["target"].apply(lambda x: ndcg_at_k(x, k)).mean()

    return res


def get_ranking_from_pred(model, X, id_cols, query_id="srch_id", item_id="prop_id"):
    id_cols.loc[:, "pred"] = model.predict(X)
    res = id_cols.sort_values([query_id, "pred"], ascending=[True, False])

    return res


def minmax_predictions(preds_list):
    temp = np.empty(shape=(preds_list[0].shape[0], len(preds_list)))

    for i, preds in enumerate(preds_list):
        temp[:, i] = (
            preds.groupby("srch_id")
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
            .to_numpy()
        )

    return temp


def objective(trial, preds: np.ndarray, test: pd.DataFrame, id_array: pd.DataFrame):
    K = preds.shape[1]
    weights = [trial.suggest_float(f"weight_{i}", 0, 1) for i in range(K)]

    id_array["pred"] = np.average(preds, weights)

    final = id_array.sort_values(["srch_id", "pred"], ascending=[True, False]).loc[
        :, ["srch_id", "prop_id"]
    ]

    score = calc_ndcg_submission(final, test, k=5)

    return score


def run_optuna_ensemble(preds, test, id_array, n_trials=50):
    study_name = "Optimize weights for ensemble prediction"
    study = optuna.create_study(direction="maximize", study_name=study_name)

    def obj(trial):
        return objective(trial, preds, test, id_array)

    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    file = f"output\\ensemble_optuna_run_{date_str()}.pickle"

    with open(file, "wb") as f:
        pickle.dump(study, f)

    return study


# Possibly faster to predict but not sure whether it is compeletly correct
# def predict_per_group2(
#     model, X: np.ndarray, id_array: np.ndarray, group: np.ndarray, columns: list
# ):
#     """
#     Predicts based on predictions per group, needs srch_id in the first column
#     """

#     N = X.shape[0]
#     K = id_array.shape[1]
#     res = np.empty(shape=(N, K), dtype=int)

#     gcum = group.cumsum()
#     gcumroll = np.roll(gcum, 1)
#     gcumroll[0] = 0

#     for g, groll in zip(gcum, gcumroll):
#         preds = model.predict(X[groll:g, :])

#         ranking = (-preds).argsort() + groll
#         ranked_items = id_array[ranking]

#         res[np.min(ranking): (np.max(ranking) + 1), :] = ranked_items

#     res = pd.DataFrame(res, columns=columns, dtype=int)

#     return res


# def predict_in_batches2(model, X, id_array, ids, g, batch_size=20000):
#     groups = g.cumsum()
#     groups_roll = np.roll(groups, 1)
#     groups_roll[0] = 0

#     n_batches = int(X.shape[0] / batch_size)

#     N = int(len(ids) / n_batches)
#     ch = chunks(ids, N)

#     res = pd.DataFrame()
#     for i, srch_ids in enumerate(tqdm(ch)):
#         if len(srch_ids) != N:
#             tempX, temp_id = X[groups_roll[i * N]:,
#                                :], id_array[groups_roll[i * N]:, :]
#         else:
#             start, end = groups_roll[i * N], groups_roll[((i + 1) * N - 1)]
#             tempX, temp_id = X[start:end, :], id_array[start:end, :]

#         group = g[(i * N): ((i + 1) * N - 1)]

#         preds = predict_per_group(
#             model, tempX, temp_id, group, columns=["srch_id", "prop_id"]
#         )

#         res = pd.concat([res, preds])

#     return res
