import joblib
import lightgbm as lgb
import optuna


class OptunaOptimization:
    def __init__(
        self,
        X_train,
        y_train,
        train_groups,
        X_val,
        y_val,
        val_groups,
        k,
        hyperparameter_config,
        n_trials=50,
        name=None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.train_groups = train_groups
        self.X_val = X_val
        self.y_val = y_val
        self.val_groups = val_groups
        self.k = k
        self.n_trials = n_trials
        self.cfg = hyperparameter_config
        self.name = name

    def _objective(
        self,
        trial,
        X_train,
        y_train,
        train_groups,
        X_val,
        y_val,
        val_groups,
        k,
        log: bool = False,
    ):
        # Initialize the model with the trial hyperparameters
        self.cfg.set_trial(trial=trial)
        self.cfg.init_params()

        # Get parameters
        params = self.cfg.get_params()
        model = self.cfg.get_model()

        # Set static and dynamic model parameters
        model.set_params(**params.get("static"))
        model.set_params(**params.get("dynamic"))

        # Train model
        model.fit(
            X=X_train,
            y=y_train,
            group=train_groups,
            eval_at=[k],
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
        )

        # Retrieve best NDCG@k score
        score = model.best_score_["valid_0"][f"ndcg@{k}"]

        return score

    def run(self):
        # Create optuna study object
        self.study = optuna.create_study(
            direction="maximize",
            study_name=self.name,
        )

        # Optimize model parameters
        self.study.optimize(
            lambda trial: self._objective(
                trial,
                self.X_train,
                self.y_train,
                self.train_groups,
                self.X_val,
                self.y_val,
                self.val_groups,
                self.k,
            ),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        return self.study

    def save_study_csv(self, path="study.csv"):
        if path is None:
            raise ValueError("Path must be specified")

        self.study.trials_dataframe().to_csv(path, index=False)

        return self.study

    def save_study_lib(self, path="study.lib"):
        if path is None:
            raise ValueError("Path must be specified")

        joblib.dump(self.study, path)

        return self.study


class HyperparameterConfig:
    def __init__(self, model) -> None:
        self.random_state = 2023
        self.model = model

    def set_trial(self, trial: optuna.trial.Trial) -> None:
        self.trial = trial

    def get_params(self, key=None) -> dict:
        if key:
            return self.params.get(key, None)
        else:
            return self.params

    def get_model(self) -> object:
        return self.model

    def get_trial(self) -> optuna.trial.Trial:
        return self.trial


class LGBMRankerConfig(HyperparameterConfig):
    def __init__(self, model=lgb.LGBMRanker()) -> None:
        super().__init__(model)
        self.params = {
            "static": {
                "n_jobs": -1,
                "objective": "lambdarank",
                "n_estimators": 300,
                "learning_rate": 0.05,
            }
        }

    def init_params(self):
        # Check if trial is set
        if self.trial is None:
            raise ValueError("Trial is not set. Please set trial first.")

        # Set model hyperparameter that are te be optimized
        self.params["dynamic"] = {
            "num_leaves": self.trial.suggest_int("num_leaves", 15, 1500),
            "max_depth": self.trial.suggest_int("max_depth", -1, 15),
            "min_data_in_leaf": self.trial.suggest_int(
                "min_data_in_leaf", 200, 10000, step=100
            ),
            "reg_alpha": self.trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": self.trial.suggest_float("reg_lambda", 0, 10),
            # "min_gain_to_split": self.trial.suggest_float("min_gain_to_split", 0, 15),
            # "subsample": self.trial.suggest_float("subsample", 0.2, 1),
            #             "bagging_freq": self.trial.suggest_categorical("bagging_freq", [1]),
            "colsample_bytree": self.trial.suggest_float("colsample_bytree", 0.2, 1),
        }


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
            #    'booking_bool_encoded',
            #    'click_bool_encoded',
            #    'target_encoded',
            #    'position_encoded'
        ]

    def get_cols(self) -> list:
        return self.usecols
