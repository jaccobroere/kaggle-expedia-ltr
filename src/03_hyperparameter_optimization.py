import optuna
import pandas as pd
import lightgbm as lgb
import pickle
import os
import logging
from utils.config.hyperparameters import HyperparameterConfig, UseColsConfig
import configparser
import datetime as dt

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

# Change working directory
os.chdir(ROOT_DIR)

# Setup logging
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename=f"{LOG_DIR}/{timestamp}_tuning_hyperparameters.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger()
lgb.register_logger(logger)


def objective(trial, X_train, y_train, group, X_val, y_val, eval_group, k=5):
    params = HyperparameterConfig(trial=trial).get_params()
    static_params = {
        "n_jobs": -1,
        "objective": "lambdarank",
        "n_estimators": 300,
        "learning_rate": 0.05,
    }

    model = lgb.LGBMRanker(
        **params,
        **static_params,
    )

    model.fit(
        X=X_train,
        y=y_train,
        group=group,
        eval_at=[k],
        eval_set=[(X_val, y_val)],
        eval_group=[eval_group],
    )

    return model.best_score_["valid_0"][f"ndcg@{k}"]


def run_optimization(
    X_train,
    y_train,
    group,
    X_val,
    y_val,
    eval_group,
    n_trials=50,
    k=5,
    name="LGBMRanker Hyperparameter Optimization",
    save=True,
):
    study_name = "LGBMRanker Hyperparameter optimization"
    study = optuna.create_study(
        direction="maximize", study_name=study_name, sampler=optuna.samplers.TPESampler
    )

    def obj(trial):
        return objective(trial, X_train, y_train, group, X_val, y_val, eval_group, k)

    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    if save:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{MODEL_DIR}/{timestamp}_{name}.pkl", "wb") as f:
            pickle.dump(study, f)

    return study


def main():
    train = pd.read_parquet(TRAIN)
    validation = pd.read_parquet(VALIDATION)

    train_groups = train.groupby("srch_id").size().to_numpy()
    val_gropus = validation.groupby("srch_id").size().to_numpy()

    target = "target"

    cols = UseColsConfig().get_cols()

    study = run_optimization(
        X_train=train[cols],
        y_train=train[target],
        group=train_groups,
        X_val=validation[cols],
        y_val=validation[target],
        eval_group=val_gropus,
        n_trials=20,
        k=5,
        name="LGBMRanker Hyperparameter Optimization",
    )

    return study


if __name__ == "__main__":
    main()
