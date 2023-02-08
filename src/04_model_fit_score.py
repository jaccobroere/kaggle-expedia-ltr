import configparser
import datetime as dt
import logging
import os

import lightgbm as lgb
import pandas as pd

from utils.config.hyperparameters import LGBMRankerConfig, UseColsConfig
from utils.helper_functions import (
    date_str,
    get_ranking_from_model,
    parse_best_params_from_csv,
)

# Read config file
config = configparser.ConfigParser()
config.read("config.ini")

# Read parameters from config file
LOG_DIR = config["LOGGING"]["LOG_DIR"]
ROOT_DIR = config["FILES"]["ROOT_DIR"]
TRAIN = config["DATA"]["CURATED_TRAINING_DATA"]
VALIDATION = config["DATA"]["CURATED_VALIDATION_DATA"]
TEST = config["DATA"]["CURATED_TEST_DATA"]
LOG_DIR = config["LOGGING"]["LOG_DIR"]
MODEL_DIR = config["MODELS"]["MODEL_DIR"]
STUDY = config["MODELS"]["STUDY"]
STUDY_CSV = config["MODELS"]["STUDY_CSV"]

# Change working directory
os.chdir(ROOT_DIR)

# Setup logging
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"{LOG_DIR}/{timestamp}_training_model.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger()
lgb.register_logger(logger)


def train():
    # Read data and parameters
    train = pd.read_parquet(TRAIN)
    validation = pd.read_parquet(VALIDATION)

    # Setup model
    best_params = parse_best_params_from_csv(STUDY_CSV)
    cfg = LGBMRankerConfig()
    model = cfg.get_model()
    model.set_params(**best_params, **cfg.get_params("static"))

    # Choose features and target variable
    target = "target"
    cols = UseColsConfig().get_cols()

    # Combine train and validation data
    train = pd.concat([train, validation], axis=0)

    # Fit model using train and validation data
    model.fit(
        train[cols],
        train[target],
        group=train.groupby("srch_id").size().to_numpy(),
        eval_at=[38],
    )

    return model


def score(model):
    # Read test data and parameters
    test = pd.read_parquet(TEST)

    # Choose features
    cols = UseColsConfig().get_cols()

    # Get the rankings from the predictions
    rankings = get_ranking_from_model(model, test, cols)

    return rankings


def main():
    # Train model
    model = train()

    # Score model and rename columns to match submission format
    rankings = score(model)
    rankings.rename(
        columns={"srch_id": "SearchId", "prop_id": "PropertyId"}, inplace=True
    )

    # Save the rankings to a csv file
    rankings.to_csv(f"{MODEL_DIR}/{date_str()}_rankings.csv", index=False)


if __name__ == "__main__":
    main()
