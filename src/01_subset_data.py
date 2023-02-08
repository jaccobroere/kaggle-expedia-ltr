from utils.data_splitter import subset_index, read_columns
import logging
import configparser
import os
from dask import dataframe as dd
import datetime as dt


def main():
    # Load config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Read config file
    ROOT_DIR = config["FILES"]["ROOT_DIR"]
    LOG_DIR = config["LOGGING"]["LOG_DIR"]

    RAW_TRAIN = config["DATA"]["RAW_TRAINING_DATA"]
    RAW_TEST = config["DATA"]["RAW_TEST_DATA"]

    TRAIN = config["DATA"]["SUBSET_TRAINING_DATA"]
    VALIDATION = config["DATA"]["SUBSET_VALIDATION_DATA"]
    TEST = config["DATA"]["SUBSET_TEST_DATA"]

    TRAIN_SIZE = config.getfloat("PARAMETERS", "TRAIN_SUBSET_PERC")
    VAL_SIZE = config.getfloat("PARAMETERS", "VALIDATION_SUBSET_PERC")
    TEST_SIZE = config.getfloat("PARAMETERS", "TEST_SUBSET_PERC")

    # Change working directory
    os.chdir(ROOT_DIR)

    # Setup logging
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=f"{LOG_DIR}/{timestamp}_subset_data.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Split the training data to a smaller set to decrease training time
    logging.info("Reading training data ...")
    train_id = read_columns(RAW_TRAIN, ["srch_id"])

    logging.info("Subsetting training and testing data ...")
    idx, val_idx = subset_index(
        train_id, col="srch_id", size=TRAIN_SIZE, val_size=VAL_SIZE
    )

    test_id = read_columns(RAW_TEST, ["srch_id"])
    test_idx = subset_index(test_id, col="srch_id", size=TEST_SIZE)

    # Read the training data, subset it and save to parquet
    logging.info("Saving subset of training and test data to .parquet ...")
    train = dd.read_csv(RAW_TRAIN)
    test = dd.read_csv(RAW_TEST)

    train.loc[lambda x: x.index.isin(idx)].to_parquet(TRAIN)
    train.loc[lambda x: x.index.isin(val_idx)].to_parquet(VALIDATION)
    test.loc[lambda x: x.index.isin(test_idx)].to_parquet(TEST)

    logging.info("Done!")


if __name__ == "__main__":
    main()
