import configparser
import datetime as dt
import logging
import os

import lightgbm as lgb
import pandas as pd

from utils.config.hyperparameters import (
    LGBMRankerConfig,
    OptunaOptimization,
    UseColsConfig,
)
from utils.helper_functions import date_str

# Read config file
config = configparser.ConfigParser()
config.optionxform = str
config.read("config.ini")

# Read parameters from config file
LOG_DIR = config["LOGGING"]["LOG_DIR"]
ROOT_DIR = config["FILES"]["ROOT_DIR"]
TRAIN = config["DATA"]["CURATED_TRAINING_DATA"]
VALIDATION = config["DATA"]["CURATED_VALIDATION_DATA"]
TEST = config["DATA"]["CURATED_TEST_DATA"]
LOG_DIR = config["LOGGING"]["LOG_DIR"]
MODEL_DIR = config["MODELS"]["MODEL_DIR"]

# Change working directory
os.chdir(ROOT_DIR)

# Setup logging
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"{LOG_DIR}/{timestamp}_tuning_hyperparameters.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger()
lgb.register_logger(logger)


def main():
    train = pd.read_parquet(TRAIN)
    validation = pd.read_parquet(VALIDATION)

    train_groups = train.groupby("srch_id").size().to_numpy()
    val_gropus = validation.groupby("srch_id").size().to_numpy()

    target = "target"
    cols = UseColsConfig().get_cols()

    optimizer = OptunaOptimization(
        X_train=train[cols],
        y_train=train[target],
        train_groups=train_groups,
        X_val=validation[cols],
        y_val=validation[target],
        val_groups=val_gropus,
        k=38,
        hyperparameter_config=LGBMRankerConfig(),
        n_trials=50,
        name="LGBMRanker Hyperparameter Optimization",
    )

    study = optimizer.run()

    logger.info("Optimization complete! \nBest hyperparameters:")
    logger.info(study.best_params)

    # Save study objects
    fname = f"{MODEL_DIR}/{date_str()}_study_{optimizer.name}.lib"
    optimizer.save_study_csv(path=f"{fname}.csv")
    optimizer.save_study_lib(f"{fname}.lib")

    # Update paths in config file
    config.set("MODELS", "STUDY", f"{fname}.lib")
    config.set("MODELS", "STUDY_CSV", f"{fname}.csv")
    config.write(open("config.ini", "w"))

    return study


if __name__ == "__main__":
    main()
