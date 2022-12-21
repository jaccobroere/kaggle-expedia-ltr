import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def LinModel(df):
    """
    Returns a Linear Regression Model Fit
    """

    y = df[['prop_starrating']].values
    X = df[['price_usd']].values
    return LinearRegression().fit(X, y)


def ModelperCountry(df):
    """
    Returns a pd.series of linear regressions fits. Each element is linear regression fit: starrating = b0 + b1 price_usd per country.
    """

    series = df.iloc[np.where(df["prop_starrating"] != 0)].groupby(
        'prop_country_id').apply(LinModel)
    return series


def ImputeStarrating0(df, series):
    """
    Returns a dataframe with the prop_starrating column being imputed for the 0 values with predictions using linear regression 
    """

    for i in np.where(df["prop_starrating"] == 0)[0]:
        if df.iloc[i]["prop_country_id"] in series.index:
            df.at[i, "prop_starrating"] = series[df.iloc[i]
                                                 ["prop_country_id"]].predict([[df.iloc[i]["price_usd"]]])
        else:
            df.at[i, "prop_starrating"] = 0

    # if the starrating is below 0 or above 5 then set equal to the bound it exceeds.
    df["prop_starrating"] = df["prop_starrating"].apply(
        lambda x: 5 if x > 5 else (0 if x < 0 else x))
    return df.copy()


if __name__ == "__main__":

    df = pd.read_csv("training_set_VU_DM.csv")      # Load data
    dfcopy = df.copy()                              # Make a copy
    df = ImputeStarrating0(df, ModelperCountry(df))  # Impute data
