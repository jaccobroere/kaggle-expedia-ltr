# %%

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
import sweetviz as sv
from imblearn.under_sampling import RandomUnderSampler
import datetime as dt
# from model_functions import *
from sklearn.linear_model import LinearRegression


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
        def func(x): return x.fillna(x.median())

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
    return dt.datetime.now().strftime("%m-%d_%H%M")


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
