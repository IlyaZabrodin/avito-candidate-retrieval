"""Признаки пар «запрос — кандидат» для модели отбора."""
import numpy as np
import pandas as pd

from src.retrieval import CharIndex, TextIndex
from src.stats import LocationStats, MicrocatClassifier, TextHistory

RATING_FILTER = "Рейтинг пользователя"


def build_features(
    pool: pd.DataFrame,
    queries: pd.DataFrame,
    items: pd.DataFrame,
    index: TextIndex,
    char_index: CharIndex,
    loc_stats: LocationStats,
    microcat: MicrocatClassifier,
    microcat_proba: np.ndarray,
    history: TextHistory,
) -> pd.DataFrame:
    """Признаки для всех пар пула. В queries нужны леммы текста (q_lem) и фильтров (f_lem).

    Пар на порядки больше, чем запросов и объявлений, поэтому всё, что зависит
    только от запроса или только от объявления, считаем один раз и раздаём по индексу.
    """
    qi, ii = pool["query_idx"].to_numpy(), pool["item_idx"].to_numpy()
    f = pool.copy()

    # насколько кандидат сильнее остальных кандидатов того же запроса
    by_query = f.groupby("query_idx")
    for col in ["bm25_all", "bm25_title", "dense"]:
        f[f"{col}_rel"] = f[col] / by_query[col].transform("max").replace(0, np.nan)
        f[f"{col}_rank"] = by_query[col].rank(ascending=False, method="first")
    f["char_title"] = char_index.pair_cosine(queries["text"].tolist(), qi, ii)

    # локация
    s_loc = queries["search_location_id"].to_numpy()[qi]
    i_loc = items["item_location_id"].to_numpy()[ii]
    f["loc_prob"] = loc_stats.prob(s_loc, i_loc)
    f["same_loc"] = (s_loc == i_loc).astype(np.int8)
    f["dist_km"] = loc_stats.distance_km(
        s_loc, items["item_latitude"].to_numpy()[ii], items["item_longitude"].to_numpy()[ii]
    )
    f["loc_searches"] = np.log1p(loc_stats.search_total.reindex(s_loc).fillna(0).to_numpy())

    # подкатегория: модель по тексту и точная история текста
    mc = items["item_microcat_id"].to_numpy()[ii]
    f["mc_prob"] = microcat.pair_prob(microcat_proba, qi, mc)
    f["mc_prob_rel"] = f["mc_prob"] / microcat_proba.max(axis=1)[qi]
    f["mc_hist"] = history.share(queries["text"], qi, mc)
    f["text_seen"] = np.log1p(history.count(queries["text"]))[qi]

    # фильтры поиска: доля слов фильтра в параметрах объявления и фильтр по рейтингу
    has_filter = queries["f_lem"].ne("").to_numpy()
    _, filter_cov = index.pair_scores(index.query_matrix(queries["f_lem"].tolist()), qi, ii, "params")
    f["filter_cov"] = np.where(has_filter[qi], filter_cov, np.nan)
    rating_filter = queries["search_infm_params_text"].str.contains(RATING_FILTER, regex=False).to_numpy()
    rating_ok = (items["item_rating"].fillna(0).to_numpy() >= 4).astype(np.float32)
    f["rating_ok"] = np.where(rating_filter[qi], rating_ok[ii], np.nan)

    # объявление
    item_feats = pd.DataFrame(
        {
            "rating": items["item_rating"].to_numpy(),
            "reviews": np.log1p(items["item_rating_reviews_count"].to_numpy()),
            "price": np.log1p(items["item_price"].to_numpy()),
            "phone_hidden": items["item_is_phone_hidden"].to_numpy().astype(np.int8),
            "msg_forbidden": items["item_is_message_forbidden"].to_numpy().astype(np.int8),
            "is_services": (items["item_category_id"].to_numpy() == 114).astype(np.int8),
            "title_len": items["item_title_raw"].str.len().to_numpy(),
            "desc_len": np.log1p(items["item_description_raw"].str.len().to_numpy()),
        }
    )
    for col in item_feats:
        f[col] = item_feats[col].to_numpy()[ii]

    # запрос
    f["q_terms"] = queries["q_lem"].str.split().str.len().to_numpy()[qi]
    f["has_filter"] = has_filter.astype(np.int8)[qi]
    f["cat_all"] = (queries["search_category"].to_numpy() == 0).astype(np.int8)[qi]
    return f


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("query_idx", "item_idx", "label")]
