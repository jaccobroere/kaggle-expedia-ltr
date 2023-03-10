# %%
import os
import pandas as pd
import numpy as np
import configparser
import logging
import tqdm
import datetime as dt
from dask import dataframe as dd
import warnings

from utils.helper_functions import (
    target_encoding_values,
    enrich_target_encoding,
    null_impute_value,
    add_target,
    add_norm_features_for_value,
    impute_over_group,
)


# %%
def clean_impute_raw(df: pd.DataFrame):
    """
    This can be performed using test and train combined
    """
    # Calculate the mismatching between historical prices of user and the current to minimize the problem of missing data
    df["visitor_hist_price_diff"] = np.abs(df["visitor_hist_adr_usd"] - df["price_usd"])
    df["visitor_hist_starrating_diff"] = np.abs(
        df["visitor_hist_starrating"] - df["prop_starrating"]
    )
    df["srch_query_affinity_score"] = np.exp(df["srch_query_affinity_score"])
    df["mean_distance_to_other_prop"] = df.groupby("srch_id")[
        "orig_destination_distance"
    ].transform(lambda x: np.abs(x - x.mean()))

    df = impute_over_group(
        df,
        group_cols="srch_id",
        columns_to_impute="mean_distance_to_other_prop",
        func=lambda x: x.fillna(x.max()),
    )

    # Impute values with 0
    df = null_impute_value(
        df,
        [
            "prop_review_score",
            "srch_query_affinity_score",
            "visitor_hist_price_diff",
            "visitor_hist_starrating_diff",
            "mean_distance_to_other_prop",
            "comp1_rate",
            "comp1_inv",
            "comp1_rate_percent_diff",
            "comp2_rate",
            "comp2_inv",
            "comp2_rate_percent_diff",
            "comp3_rate",
            "comp3_inv",
            "comp3_rate_percent_diff",
            "comp4_rate",
            "comp4_inv",
            "comp4_rate_percent_diff",
            "comp5_rate",
            "comp5_inv",
            "comp5_rate_percent_diff",
            "comp6_rate",
            "comp6_inv",
            "comp6_rate_percent_diff",
            "comp7_rate",
            "comp7_inv",
            "comp7_rate_percent_diff",
            "comp8_rate",
            "comp8_inv",
            "comp8_rate_percent_diff",
        ],
        value=0,
    )

    # Impute origin destination distance using similar location proxys
    df = impute_over_group(
        df,
        ["visitor_location_country_id", "prop_country_id"],
        "orig_destination_distance",
    )
    df = impute_over_group(df, ["prop_country_id"], "orig_destination_distance")
    df["orig_destination_distance"] = df["orig_destination_distance"].transform(
        lambda x: x.fillna(x.median())
    )

    df = impute_over_group(
        df,
        group_cols="prop_country_id",
        columns_to_impute="prop_location_score2",
        func=lambda x: x.fillna(np.quantile(x, 0.25)),
    )

    df["site_id"] = df["site_id"].astype(int)

    return df


def transform_training_set(df: pd.DataFrame):
    df["position"] = (
        df.loc[:, ["position", "srch_id"]]
        .groupby("srch_id")
        .transform(lambda x: 1 - (x - x.min()) / (x.max() - x.min()))
    )

    df = add_target(df)

    return df


def add_normalized_features(df: pd.DataFrame):
    df = add_norm_features_for_value(df, value_col="price_usd")
    df = add_norm_features_for_value(df, value_col="prop_log_historical_price")

    return df


def add_engineered_features(df: pd.DataFrame):
    df["fee_per_person"] = (
        df.loc[:, "price_usd"]
        * df.loc[:, "srch_room_count"]
        / (df.loc[:, "srch_adults_count"] + df.loc[:, "srch_children_count"])
    )

    df["score2ma"] = (
        df.loc[:, "prop_location_score2"] * df.loc[:, "srch_query_affinity_score"]
    )

    group = df.groupby("srch_id")
    df["rank_price"] = group["price_usd"].rank("dense")
    df["rank_location_score1"] = group["prop_location_score1"].rank("dense")
    df["rank_location_score2"] = group["prop_location_score2"].rank("dense")
    df["rank_starrating"] = group["prop_starrating"].rank("dense")
    df["rank_review_score"] = group["prop_review_score"].rank("dense")

    group = df.groupby("prop_id")
    df["avg_price_prop_id"] = group["price_usd"].transform(lambda x: x.mean())
    df["median_price_prop_id"] = group["price_usd"].transform(lambda x: x.median())
    df["std_price_prop_id"] = group["price_usd"].transform(lambda x: x.std())
    df["avg_location_score2_prop_id"] = group["prop_location_score2"].transform(
        lambda x: x.mean()
    )
    df["median_location_score2_prop_id"] = group["prop_location_score2"].transform(
        lambda x: x.median()
    )
    df["std_location_score2_prop_id"] = group["prop_location_score2"].transform(
        lambda x: x.std()
    )
    df["avg_srch_booking_window_prop_id"] = group["srch_booking_window"].transform(
        lambda x: x.mean()
    )
    df["median_srch_booking_window_prop_id"] = group["srch_booking_window"].transform(
        lambda x: x.median()
    )
    df["std_srch_booking_window_prop_id"] = group["srch_booking_window"].transform(
        lambda x: x.std()
    )
    df["avg_srch_adults_count_prop_id"] = group["srch_adults_count"].transform(
        lambda x: x.mean()
    )
    df["median_srch_adults_count_prop_id"] = group["srch_adults_count"].transform(
        lambda x: x.median()
    )
    df["std_srch_adults_count_prop_id"] = group["srch_adults_count"].transform(
        lambda x: x.std()
    )
    df["avg_srch_saturday_night_bool_prop_id"] = group[
        "srch_saturday_night_bool"
    ].transform(lambda x: x.mean())
    df["median_srch_saturday_night_bool_prop_id"] = group[
        "srch_saturday_night_bool"
    ].transform(lambda x: x.median())
    df["std_srch_saturday_night_bool_prop_id"] = group[
        "srch_saturday_night_bool"
    ].transform(lambda x: x.std())
    df["avg_srch_room_count_prop_id"] = group["srch_room_count"].transform(
        lambda x: x.mean()
    )
    df["median_srch_room_count_prop_id"] = group["srch_room_count"].transform(
        lambda x: x.median()
    )
    df["std_srch_room_count_prop_id"] = group["srch_room_count"].transform(
        lambda x: x.std()
    )
    df["avg_srch_children_count_prop_id"] = group["srch_children_count"].transform(
        lambda x: x.mean()
    )
    df["median_srch_children_count_prop_id"] = group["srch_children_count"].transform(
        lambda x: x.median()
    )
    df["std_srch_children_count_prop_id"] = group["srch_children_count"].transform(
        lambda x: x.std()
    )
    df["avg_srch_length_of_stay_prop_id"] = group["srch_length_of_stay"].transform(
        lambda x: x.mean()
    )
    df["median_srch_length_of_stay_prop_id"] = group["srch_length_of_stay"].transform(
        lambda x: x.median()
    )
    df["std_srch_length_of_stay_prop_id"] = group["srch_length_of_stay"].transform(
        lambda x: x.std()
    )
    df["avg_srch_query_affinity_score_prop_id"] = group[
        "srch_query_affinity_score"
    ].transform(lambda x: x.mean())
    df["median_srch_query_affinity_score_prop_id"] = group[
        "srch_query_affinity_score"
    ].transform(lambda x: x.median())
    df["std_srch_query_affinity_score_prop_id"] = group[
        "srch_query_affinity_score"
    ].transform(lambda x: x.std())

    return df


def add_target_encoded_features(to_enrich: pd.DataFrame, train: pd.DataFrame):
    encoded_prop_id = target_encoding_values(
        train,
        "prop_id",
        ["booking_bool", "click_bool", "target", "position"],
        agg_func="mean",
    )

    to_enrich = enrich_target_encoding(to_enrich, encoded_prop_id, group_col="prop_id")

    return to_enrich


def impute_last_missing_values(df: pd.DataFrame):
    df = null_impute_value(
        df,
        [
            "booking_bool_encoded",
            "click_bool_encoded",
            "target_encoded",
            "position_encoded",
        ],
        value=0,
    )

    return df


def run_pipe(train, test, val):
    for df, name in zip([train, test, val], ["train", "test", "val"]):
        df["DATA_CAT"] = name

    pbar = tqdm.tqdm(total=100)
    full = pd.concat([train, val, test], ignore_index=True, join="outer", copy=False)

    pbar.set_description("Imputing data")
    full = clean_impute_raw(full)
    pbar.update(25)

    pbar.set_description("Adding normalized features")
    full = add_normalized_features(full)
    pbar.update(25)

    pbar.set_description("Adding engineered features")
    full = add_engineered_features(full)
    pbar.update(25)

    train, val, test = (
        full.loc[full.DATA_CAT == "train", :].drop(columns="DATA_CAT"),
        full.loc[full.DATA_CAT == "val", :].drop(columns="DATA_CAT"),
        full.loc[full.DATA_CAT == "test", :].drop(columns="DATA_CAT"),
    )

    # pbar.set_description("Adding target encoded features")
    # train = add_target_encoded_features(train, train)
    # test = add_target_encoded_features(test, train)

    # train = impute_last_missing_values(train)
    # val = impute_last_missing_values(val)
    # test = impute_last_missing_values(test)

    columns_to_drop = [
        "position",
        "click_bool",
        "gross_bookings_usd",
        "booking_bool",
    ]

    test.drop(
        columns=columns_to_drop,
        inplace=True,
    )

    train = add_target(train, target_col="target")
    val = add_target(val, target_col="target")

    pbar.update(25)

    return train, val, test


def main():
    # Load config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Read config file
    ROOT_DIR = config["FILES"]["ROOT_DIR"]
    TRAIN = config["DATA"]["SUBSET_TRAINING_DATA"]
    VALIDATION = config["DATA"]["SUBSET_VALIDATION_DATA"]
    TEST = config["DATA"]["SUBSET_TEST_DATA"]
    LOG_DIR = config["LOGGING"]["LOG_DIR"]

    # Change working directory
    os.chdir(ROOT_DIR)

    # Setup logging
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=f"{LOG_DIR}/{timestamp}_imputation_feature_engineering.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # # Load training data and add target to train data
    logging.info("Loading data...")
    train = dd.read_parquet(
        TRAIN,
        infer_datetime_format=True,
        parse_dates=[2],
    ).compute()
    val = dd.read_parquet(
        VALIDATION,
        infer_datetime_format=True,
        parse_dates=[2],
    ).compute()
    test = dd.read_parquet(TEST, infer_datetime_format=True, parse_dates=[2]).compute()
    logging.info("Data loading succesful!")

    logging.info("Running data pipeline...")
    train, val, test = run_pipe(train=train, test=test, val=val)
    logging.info("Data pipeline succesful!")

    # # Write the curated data to disk
    train.to_parquet(r"data/curated_train.parquet", index=False)
    logging.info("Saved curated train data!")

    val.to_parquet(r"data/curated_val.parquet", index=False)
    logging.info("Saved curated val data!")

    test.to_parquet(r"data/curated_test.parquet", index=False)
    logging.info("Saved curated test data!")


# %%
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
