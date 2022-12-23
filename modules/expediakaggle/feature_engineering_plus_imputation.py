# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:54:43 2022

@author: ruben
"""

import pandas as pd
import numpy as np


def data_imputation(df: pd.DataFrame):
    """
    Imputes missing values
    """
    df["prop_location_score2"] = df["prop_location_score2"].fillna(
        trainset.groupby("prop_id")["prop_location_score2"].transform("mean")
    )
    df["prop_review_score"] = df["prop_review_score"].replace(np.nan, 0)

    sub = df[df["random_bool"] == 1]
    sub["newposition"] = sub.groupby("srch_id")["position"].transform(
        lambda x: x.mean()
    )
    df.loc[trainset["random_bool"] == 1, "position"] = sub["newposition"]

    # Weet niet of het volgende de beste uitkomst heeft
    df["prop_log_historical_price"] = df["prop_log_historical_price"].replace(0, np.nan)
    df["prop_log_historical_price"] = df["prop_log_historical_price"].fillna(
        df.groupby("prop_id")["prop_log_historical_price"].transform("mean")
    )

    return df.copy()


def featurize(df: pd.DataFrame):
    """
    Adds new features to dataframe
    """

    df["hist_price_diff"] = (
        np.exp(df.loc[:, "prop_log_historical_price"]) - df.loc[:, "price_usd"]
    )
    df["diff_price"] = df.loc[:, "visitor_hist_adr_usd"] - trainset.loc[:, "price_usd"]
    df["diff_starrating"] = (
        df.loc[:, "visitor_hist_starrating"] - df.loc[:, "prop_starrating"]
    )
    df["fee_per_person"] = (
        df.loc[:, "price_usd"]
        * df.loc[:, "srch_room_count"]
        / (df.loc[:, "srch_adults_count"] + df.loc[:, "srch_children_count"])
    )
    df["score2ma"] = (
        df.loc[:, "prop_location_score2"] * df.loc[:, "srch_query_affinity_score"]
    )
    df["total_price"] = df.loc[:, "price_usd"] * df.loc[:, "srch_room_count"]
    df["CTR"] = df.groupby("prop_id")["click_bool"].transform(lambda x: x.mean())
    df["BR"] = df.groupby("prop_id")["booking_bool"].transform(lambda x: x.mean())

    return df.copy()
