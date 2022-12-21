import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
import pickle
from .preprocessing import date_str


def predict_per_group(
    model, X: np.ndarray, id_array: np.ndarray, group: np.ndarray, columns: list
):
    """
    Predicts based on predictions per group, needs srch_id in the first column
    """

    N = X.shape[0]
    K = id_array.shape[1]
    res = np.empty(shape=(N, K), dtype=int)

    idx = 0
    for i, srch_id in enumerate(np.unique(id_array[:, 0])):
        mask = id_array[:, 0] == srch_id
        preds = model.predict(X[mask])

        ranking = (-preds).argsort() + idx
        ranked_items = id_array[ranking]

        res[np.min(ranking): (np.max(ranking) + 1), :] = ranked_items

        idx += group[i]

    res = pd.DataFrame(res, columns=columns, dtype=int)

    return res


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def predict_in_batches(model, df, cols, id_cols, batch_size=20000):
    n_batches = int(df.shape[0] / batch_size)
    res = pd.DataFrame()

    ids = list(df["srch_id"].unique())

    ch = chunks(ids, int(len(ids) / n_batches))

    for srch_ids in tqdm(ch):
        temp = df.loc[df["srch_id"].isin(srch_ids), :].copy()

        X, id_array = temp[cols].to_numpy(), temp[id_cols].to_numpy()
        group = temp.groupby("srch_id").size().to_numpy()
        preds = predict_per_group(
            model, X, id_array, group, columns=["srch_id", "prop_id"]
        )

        res = pd.concat([res, preds])

    return res


def ndcg_at_k(target, k=5):
    k = min([k, target.shape[0]])
    idx = np.log2(np.array([i + 2 for i in range(k)]))
    dcg = (target[:k] / idx).sum()

    idcg = (np.sort(target)[::-1][:k] / idx).sum()

    return dcg / idcg


def calc_ndcg_submission(submission, df, k=5):
    ranking = pd.merge(
        left=submission,
        right=df,
        left_on=["srch_id", "prop_id"],
        right_on=["srch_id", "prop_id"],
    )[["srch_id", "prop_id", "target"]]

    res = ranking.groupby("srch_id")["target"].apply(
        lambda x: ndcg_at_k(x, k)).mean()

    return res


def get_ranking_from_pred(model, X, id_cols, query_id="srch_id", item_id="prop_id"):
    id_cols.loc[:, "pred"] = model.predict(X)
    res = id_cols.sort_values([query_id, "pred"], ascending=[True, False])

    return res


def minmax_predictions(preds_list):
    temp = np.empty(shape=(preds_list[0].shape[0], len(preds_list)))

    for i, preds in enumerate(preds_list):
        temp[:, i] = preds.groupby("srch_id").transform(
            lambda x: (x - x.min()) / (x.max() - x.min())).to_numpy()

    return temp


def objective(trial, preds: np.ndarray, test: pd.DataFrame, id_array: pd.DataFrame):
    K = preds.shape[1]
    weights = [trial.suggest_float(f"weight_{i}", 0, 1) for i in range(K)]

    id_array["pred"] = np.average(preds, weights)

    final = id_array.sort_values(["srch_id", "pred"], ascending=[
                                 True, False]).loc[:, ["srch_id", "prop_id"]]

    score = calc_ndcg_submission(final, test, k=5)

    return score


def run_optuna_ensemble(preds, test, id_array, n_trials=50):
    study_name = 'Optimize weights for ensemble prediction'
    study = optuna.create_study(direction="maximize", study_name=study_name)

    def obj(trial):
        return objective(trial, preds, test, id_array)

    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)

    file = f"output\\ensemble_optuna_run_{date_str()}.pickle"

    with open(file, "wb") as f:
        pickle.dump(study, f)

    return study


# Possibly faster to predict but not sure whether it is compeletly correct
# def predict_per_group2(
#     model, X: np.ndarray, id_array: np.ndarray, group: np.ndarray, columns: list
# ):
#     """
#     Predicts based on predictions per group, needs srch_id in the first column
#     """

#     N = X.shape[0]
#     K = id_array.shape[1]
#     res = np.empty(shape=(N, K), dtype=int)

#     gcum = group.cumsum()
#     gcumroll = np.roll(gcum, 1)
#     gcumroll[0] = 0

#     for g, groll in zip(gcum, gcumroll):
#         preds = model.predict(X[groll:g, :])

#         ranking = (-preds).argsort() + groll
#         ranked_items = id_array[ranking]

#         res[np.min(ranking): (np.max(ranking) + 1), :] = ranked_items

#     res = pd.DataFrame(res, columns=columns, dtype=int)

#     return res


# def predict_in_batches2(model, X, id_array, ids, g, batch_size=20000):
#     groups = g.cumsum()
#     groups_roll = np.roll(groups, 1)
#     groups_roll[0] = 0

#     n_batches = int(X.shape[0] / batch_size)

#     N = int(len(ids) / n_batches)
#     ch = chunks(ids, N)

#     res = pd.DataFrame()
#     for i, srch_ids in enumerate(tqdm(ch)):
#         if len(srch_ids) != N:
#             tempX, temp_id = X[groups_roll[i * N]:,
#                                :], id_array[groups_roll[i * N]:, :]
#         else:
#             start, end = groups_roll[i * N], groups_roll[((i + 1) * N - 1)]
#             tempX, temp_id = X[start:end, :], id_array[start:end, :]

#         group = g[(i * N): ((i + 1) * N - 1)]

#         preds = predict_per_group(
#             model, tempX, temp_id, group, columns=["srch_id", "prop_id"]
#         )

#         res = pd.concat([res, preds])

#     return res
