"""Загрузка данных, сборка запросов из train и разбиение для валидации."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILES = ["train.parquet", "benchmark_queries.parquet", "benchmark_items.parquet"]

QUERY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]
TEXT_COLS = ["item_title_raw", "item_description_raw", "item_infm_params_text"]


def norm_query(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .str.lower()
        .str.replace("ё", "е")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _clean_queries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["search_infm_params_text"] = df["search_infm_params_text"].fillna("")
    df["text"] = norm_query(df["search_query"])
    return df


def _clean_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df[ITEM_COLS].copy()
    # цена и координаты лежат в parquet как decimal
    for col in ["item_price", "item_latitude", "item_longitude"]:
        df[col] = df[col].astype(float)
    # цены -1, 0 и 1 — заглушки вида «цена по запросу», а не реальные значения
    df.loc[df["item_price"] <= 1, "item_price"] = np.nan
    for col in TEXT_COLS:
        df[col] = df[col].fillna("")
    return df.reset_index(drop=True)


def load_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = _clean_queries(pd.read_parquet(DATA_DIR / "benchmark_queries.parquet"))
    items = _clean_items(pd.read_parquet(DATA_DIR / "benchmark_items.parquet"))
    return queries, items


def load_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает запросы из train (с колонкой relevant) и уникальные объявления из train.

    В train нет query_id, поэтому запросом считаем уникальный набор признаков поиска:
    все объявления, выбранные при одинаковых признаках, — его релевантные ответы.
    """
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    # около 30 тысяч строк — полные дубли пар «запрос — объявление»
    train = train.drop_duplicates(QUERY_COLS + ["item_id"])
    train["search_infm_params_text"] = train["search_infm_params_text"].fillna("")

    queries = (
        train.groupby(QUERY_COLS, sort=False)["item_id"]
        .agg(list)
        .rename("relevant")
        .reset_index()
    )
    queries = _clean_queries(queries)
    queries.insert(0, "query_id", "t" + queries.index.astype(str))

    items = _clean_items(train.drop_duplicates("item_id"))
    return queries, items


def dev_split(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    bench_queries: pd.DataFrame,
    corpus_size: int,
    n_val: int,
    n_rank: int,
    seed: int = 0,
) -> dict:
    """Отложенные запросы и корпус, устроенные как бенчмарк.

    В бенчмарке все тексты запросов уникальны, а доля текстов, встречавшихся в train,
    совпадает с тем, что даёт выбор одной случайной группы на текст. Поэтому берём
    по одной группе на текст и выравниваем долю запросов без фильтров под бенчмарк.

    val — для оценки качества, rank — для обучения модели отбора кандидатов.
    fit — остальные пары «запрос — объявление» из train, на них считается вся статистика.
    """
    one_per_text = queries.sample(frac=1, random_state=seed).drop_duplicates("text")
    no_filter = one_per_text["search_infm_params_text"].eq("")
    share = bench_queries["search_infm_params_text"].eq("").mean()

    n_total = n_val + n_rank
    n_empty = round(share * n_total)
    holdout = pd.concat(
        [
            one_per_text[no_filter].sample(n_empty, random_state=seed),
            one_per_text[~no_filter].sample(n_total - n_empty, random_state=seed),
        ]
    ).sample(frac=1, random_state=seed)
    val, rank = holdout.iloc[:n_val], holdout.iloc[n_val:]

    # из fit убираем и сами отложенные группы, и все пары с их объявлениями,
    # иначе модели увидят релевантные объявления до проверки
    holdout_items = set(holdout["relevant"].explode())
    fit = (
        queries[~queries["query_id"].isin(holdout["query_id"])]
        .explode("relevant")
        .rename(columns={"relevant": "item_id"})
    )
    fit = fit[~fit["item_id"].isin(holdout_items)]

    # корпус как в бенчмарке: все релевантные плюс другие выбранные объявления до того же размера
    rng = np.random.default_rng(seed)
    is_holdout = items["item_id"].isin(holdout_items).to_numpy()
    others = np.flatnonzero(~is_holdout)
    extra = rng.choice(others, size=corpus_size - is_holdout.sum(), replace=False)
    corpus = items.iloc[np.sort(np.concatenate([np.flatnonzero(is_holdout), extra]))]

    return {
        "fit": fit.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "rank": rank.reset_index(drop=True),
        "corpus": corpus.reset_index(drop=True),
    }
