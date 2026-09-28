"""Корпус с индексами и признаки кандидатов для набора запросов."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.candidates import LocationMasks, PoolConfig, build_pool
from src.embeddings import QUERY_LEN, Encoder, query_texts
from src.features import build_features
from src.retrieval import CharIndex, TextIndex
from src.stats import LocationStats, MicrocatClassifier, TextHistory, microcat_docs
from src.text import Lemmatizer


@dataclass
class Models:
    """Всё, что обучается на fit-части train и затем без изменений применяется к бенчмарку."""

    lemmatizer: Lemmatizer
    loc_stats: LocationStats
    history: TextHistory
    microcat: MicrocatClassifier
    encoder: Encoder


class Corpus:
    """Корпус объявлений со всем, что нужно для поиска кандидатов."""

    def __init__(
        self, items: pd.DataFrame, fields: dict[str, list[str]], item_emb: np.ndarray, models: Models
    ):
        self.items = items
        self.index = TextIndex(fields)
        self.char_index = CharIndex(items["item_title_raw"].str.lower().str.replace("ё", "е").tolist())
        self.masks = LocationMasks(items["item_location_id"].to_numpy(), models.loc_stats)
        self.emb = item_emb


def candidate_features(
    queries: pd.DataFrame, corpus: Corpus, models: Models, cfg: PoolConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Пул кандидатов с признаками. Возвращает и запросы с леммами: по их порядку
    нумеруются query_idx в пуле."""
    cfg = cfg or PoolConfig()
    queries = queries.reset_index(drop=True).copy()
    queries["q_lem"] = models.lemmatizer(queries["text"])
    queries["f_lem"] = models.lemmatizer(queries["search_infm_params_text"])
    query_emb = models.encoder.encode(query_texts(queries), QUERY_LEN)

    pool = build_pool(
        corpus.index,
        queries["q_lem"].tolist(),
        queries["search_location_id"].to_numpy(),
        queries["search_category"].to_numpy(),
        corpus.items["item_category_id"].to_numpy(),
        corpus.masks,
        cfg,
        query_emb,
        corpus.emb,
    )
    proba = models.microcat.predict_proba(microcat_docs(queries["q_lem"], queries["f_lem"]))
    features = build_features(
        pool,
        queries,
        corpus.items,
        corpus.index,
        corpus.char_index,
        models.loc_stats,
        models.microcat,
        proba,
        models.history,
    )
    return features, queries
