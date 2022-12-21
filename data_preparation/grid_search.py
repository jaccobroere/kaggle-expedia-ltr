import optuna
import pandas as pd
import lightgbm as lgb
import pickle
from project_modules.preprocessing import date_str, add_target, train_val_split_group
from tqdm import tqdm
import os
import logging


def objective(trial, X_train, y_train, group, X_val, y_val, eval_group, k=5):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 1500),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        # "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        # "subsample": trial.suggest_float("subsample", 0.2, 1),
        #             "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1),
    }

    model = lgb.LGBMRanker(**params, n_jobs=-1, objective="lambdarank",
                           n_estimators=300, learning_rate=0.05)

    logging.basicConfig(
        level=logging.INFO, filename=f"logs\grid_search.log", encoding="utf-8")
    logger = logging.getLogger()
    lgb.register_logger(logger)

    model.fit(
        X=X_train,
        y=y_train,
        group=group,
        eval_at=[k],
        eval_set=[(X_val, y_val)],
        eval_group=[eval_group]
    )

    score = model.best_score_['valid_0'][f"ndcg@{k}"]

    return score


def run_optuna(X_train, y_train, group, X_val, y_val, eval_group, n_trials=50, k=5):
    study_name = 'LGBMRanker Hyperparameter optimization'
    study = optuna.create_study(direction="maximize", study_name=study_name)

    def obj(trial):
        return objective(trial, X_train, y_train, group, X_val, y_val, eval_group, k)

    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    print(f"Best value: {study.best_value: .4f}")

    with open(f"output\\optuna_run_{date_str()}.pickle", 'wb') as f:
        pickle.dump(study, f)

    return study


def main():
    # os.chdir(r"C:\Users\jacbr\OneDrive\Documenten\vu-data-mining-techniques\Assignment 2")
    os.chdir(r"C:\Users\Beheerder\Documents\vu-data-mining-techniques\Assignment 2")

    train = add_target(pd.read_csv(r"data\curated_train_dev.csv",
                       infer_datetime_format=True, parse_dates=[2]))
    test = pd.read_csv(r"data\curated_val_dev.csv",
                       infer_datetime_format=True, parse_dates=[2])

    train, _ = train_val_split_group(train, test_size=0.8)

    train_groups = train.groupby("srch_id").size().to_numpy()
    test_groups = test.groupby("srch_id").size().to_numpy()

    target = "target"
    cols = [
        "prop_starrating",
        "prop_review_score",
        "prop_brand_bool",
        "prop_location_score1",
        "prop_location_score2",
        "prop_log_historical_price",
        "price_usd",
        "promotion_flag",
        "srch_destination_id",
        "srch_length_of_stay",
        "srch_booking_window",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count",
        "srch_saturday_night_bool",
        "srch_query_affinity_score",
        "orig_destination_distance",
        "random_bool",
        # "comp1_rate",
        # "comp1_inv",
        # "comp1_rate_percent_diff",
        # "comp2_rate",
        # "comp2_inv",
        # "comp2_rate_percent_diff",
        # "comp3_rate",
        # "comp3_inv",
        # "comp3_rate_percent_diff",
        # "comp4_rate",
        # "comp4_inv",
        # "comp4_rate_percent_diff",
        # "comp5_rate",
        # "comp5_inv",
        # "comp5_rate_percent_diff",
        # "comp6_rate",
        # "comp6_inv",
        # "comp6_rate_percent_diff",
        # "comp7_rate",
        # "comp7_inv",
        # "comp7_rate_percent_diff",
        # "comp8_rate",
        # "comp8_inv",
        # "comp8_rate_percent_diff",
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "visitor_hist_price_diff",
        "visitor_hist_starrating_diff",
        "norm_price_usd_wrt_srch_id",
        "norm_price_usd_wrt_prop_id",
        "norm_price_usd_wrt_srch_destination_id",
        "month",
        "norm_price_usd_wrt_month",
        "norm_price_usd_wrt_srch_booking_window",
        "norm_price_usd_wrt_prop_country_id",
        "norm_prop_log_historical_price_wrt_srch_id",
        "norm_prop_log_historical_price_wrt_prop_id",
        "norm_prop_log_historical_price_wrt_srch_destination_id",
        "norm_prop_log_historical_price_wrt_month",
        "norm_prop_log_historical_price_wrt_srch_booking_window",
        "norm_prop_log_historical_price_wrt_prop_country_id",
        "fee_per_person",
        "score2ma",
        "rank_price",
        "rank_location_score1",
        "rank_location_score2",
        "rank_starrating",
        "rank_review_score",
        "avg_price_prop_id",
        "median_price_prop_id",
        "std_price_prop_id",
        "avg_location_score2_prop_id",
        "median_location_score2_prop_id",
        "std_location_score2_prop_id"
        #    'booking_bool_encoded', 'click_bool_encoded',
        #    'target_encoded', 'position_encoded'
    ]

    study = run_optuna(train[cols], train[target], train_groups,
                       test[cols], test[target], test_groups, n_trials=500)

    return study


if __name__ == "__main__":
    main()
