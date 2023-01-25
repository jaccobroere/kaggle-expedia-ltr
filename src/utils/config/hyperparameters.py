import optuna


class HyperparameterConfig:
    def __init__(self, trial: optuna.trial.Trial) -> None:
        self.trial = trial
        self.num_leaves = trial.suggest_int("num_leaves", 15, 1500)
        self.max_depth = trial.suggest_int("max_depth", -1, 15)
        self.min_data_in_leaf = trial.suggest_int(
            "min_data_in_leaf", 200, 10000, step=100
        )
        self.min_gain_to_split = trial.suggest_float("min_gain_to_split", 0, 15)
        self.subsample = trial.suggest_float("subsample", 0.2, 1)
        self.colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1)

    def get_params(self) -> dict:
        params = {
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_gain_to_split": self.min_gain_to_split,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }
        return params


class UseColsConfig:
    def __init__(self):
        self.usecols = [
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

    def get_cols(self) -> list:
        return self.usecols
