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

    # Set root directory
    root_dir = config["FILES"]["ROOT_DIR"]
    os.chdir(root_dir)

    # Setup logging
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=f"{config['LOGGING']['LOG_DIR']}/{timestamp}_subset_data.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Split the training data to a smaller set to decrease training time
    logging.info("Reading training data ...")
    train_id = read_columns(config["DATA"]["RAW_TRAINING_DATA"], ["srch_id"])

    logging.info("Subsetting training data ...")
    idx, val_idx = subset_index(
        train_id,
        col="srch_id",
        size=config.getfloat("PARAMETERS", "TRAIN_SUBSET_PERC"),
        val_size=config.getfloat("PARAMETERS", "VALIDATION_SUBSET_PERC"),
    )

    # Read the training data, subset it and save to parquet
    logging.info("Saving subset of training data to .parquet ...")
    train = dd.read_csv(config["DATA"]["RAW_TRAINING_DATA"])
    train.loc[lambda x: x.index.isin(idx)].to_parquet(
        config["DATA"]["SUBSET_TRAINING_DATA"]
    )
    train.loc[lambda x: x.index.isin(val_idx)].to_parquet(
        config["DATA"]["SUBSET_VALIDATION_DATA"]
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
