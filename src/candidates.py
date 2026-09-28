"""Пул кандидатов для каждого запроса и текстовые признаки пар."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.retrieval import TextIndex
from src.stats import LocationStats

FIELDS = ["title", "params", "desc", "all"]


@dataclass
class PoolConfig:
    k_loc: int = 300  # BM25 по всем полям в совместимых локациях
    k_core: int = 150  # BM25 по всем полям в ядре локации, чтобы соседи не вытесняли своих
    k_title: int = 100  # BM25 по заголовку в совместимых локациях
    k_global: int = 100  # BM25 без учёта локации: удалённые услуги, соседние регионы
    k_dense: int = 200  # эмбеддинги в совместимых локациях
    k_dense_core: int = 100  # эмбеддинги в ядре локации
    k_dense_global: int = 50  # эмбеддинги по всему корпусу: минимум 50 кандидатов есть всегда
    batch: int = 256  # запросов за один проход по корпусу


def top_k(values: np.ndarray, k: int) -> np.ndarray:
    """Позиции k наибольших значений; порядок и ничьи — детерминированно."""
    if len(values) > k:
        part = np.argpartition(-values, k - 1)[:k]
    else:
        part = np.arange(len(values))
    return part[np.lexsort((part, -values[part]))]


class LocationMasks:
    """Маски объявлений для локации поиска: все совместимые локации и ядро.

    Помнит только последнюю локацию: build_pool идёт по запросам в порядке локаций.
    """

    def __init__(self, item_loc: np.ndarray, stats: LocationStats):
        self.codes, self.uniques = pd.factorize(item_loc)
        self.stats = stats
        self._last: tuple[int, tuple[np.ndarray, np.ndarray]] | None = None

    def __call__(self, search_loc: int) -> tuple[np.ndarray, np.ndarray]:
        if self._last is None or self._last[0] != search_loc:
            compatible = self.stats.compatible.get(search_loc, set()) | {search_loc}
            core = self.stats.core.get(search_loc, set()) | {search_loc}
            self._last = (
                search_loc,
                (
                    np.isin(self.uniques, list(compatible))[self.codes],
                    np.isin(self.uniques, list(core))[self.codes],
                ),
            )
        return self._last[1]


def build_pool(
    index: TextIndex,
    query_lemmas: list[str],
    search_loc: np.ndarray,
    search_cat: np.ndarray,
    item_cat: np.ndarray,
    masks: LocationMasks,
    cfg: PoolConfig,
    query_emb: np.ndarray,
    item_emb: np.ndarray,
) -> pd.DataFrame:
    """Объединение кандидатов из всех источников с BM25 и покрытием по полям.

    Если поиск шёл в конкретной категории, кандидаты только из неё: в train
    так почти всегда. Поиск без категории (0) видит весь корпус.
    """
    order = np.lexsort((search_cat, search_loc))
    everything = np.ones(len(item_cat), dtype=bool)
    in_category = {c: item_cat == c for c in np.unique(search_cat) if c != 0}
    parts = []
    for start in range(0, len(order), cfg.batch):
        qi = order[start : start + cfg.batch]
        q = index.query_matrix([query_lemmas[i] for i in qi])
        s_all = index.scores(q, "all")
        s_title = index.scores(q, "title")
        # округление убирает шум последних знаков между устройствами
        dense = np.round(query_emb[qi] @ item_emb.T, 4)

        rows, cols = [], []
        for j, i in enumerate(qi):
            compatible, core = masks(search_loc[i])
            anywhere = in_category.get(search_cat[i], everything)
            compatible, core = compatible & anywhere, core & anywhere
            cand = []
            for scores, mask, k in [
                (s_all, compatible, cfg.k_loc),
                (s_all, core, cfg.k_core),
                (s_all, anywhere, cfg.k_global),
                (s_title, compatible, cfg.k_title),
            ]:
                lo, hi = scores.indptr[j], scores.indptr[j + 1]
                idx, val = scores.indices[lo:hi], scores.data[lo:hi]
                ok = mask[idx]
                cand.append(idx[ok][top_k(val[ok], k)])
            for mask, k in [
                (compatible, cfg.k_dense),
                (core, cfg.k_dense_core),
                (anywhere, cfg.k_dense_global),
            ]:
                allowed = np.flatnonzero(mask)
                cand.append(allowed[top_k(dense[j, allowed], k)])
            cand = np.unique(np.concatenate(cand))
            rows.append(np.full(len(cand), j))
            cols.append(cand)
        rows, cols = np.concatenate(rows), np.concatenate(cols)

        part = {"query_idx": qi[rows], "item_idx": cols}
        for field in FIELDS:
            part[f"bm25_{field}"], part[f"cov_{field}"] = index.pair_scores(q, rows, cols, field)
        part["dense"] = dense[rows, cols]
        parts.append(pd.DataFrame(part))

    pool = pd.concat(parts, ignore_index=True)
    return pool.sort_values(["query_idx", "item_idx"], kind="stable", ignore_index=True)
