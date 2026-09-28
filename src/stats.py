"""Статистики по выбранным парам из train: локации и подкатегории."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

LOC_BASE = 10**7  # id локаций меньше, так что пара (поиск, объявление) кодируется одним числом
MICROCAT_BASE = 10**8  # то же для пары (текст запроса, подкатегория)


class LocationStats:
    """Куда ведут поиски из каждой локации.

    Локации бывают разного уровня: поиск по области или по всей стране
    приводит к объявлениям из городов внутри неё, поэтому совместимость
    локаций берём из данных, а не из равенства id.
    """

    def fit(self, pairs: pd.DataFrame, core_min_prob: float = 0.05) -> "LocationStats":
        cnt = pairs.groupby(["search_location_id", "item_location_id"]).size()
        s_loc = cnt.index.get_level_values(0).to_numpy()
        i_loc = cnt.index.get_level_values(1).to_numpy()
        self.search_total = cnt.groupby(level=0).sum()
        self._keys = pd.Index(s_loc * LOC_BASE + i_loc)
        self._prob = (cnt / self.search_total.reindex(s_loc).to_numpy()).to_numpy()
        self.compatible = pd.Series(i_loc).groupby(s_loc).agg(set).to_dict()
        # ядро — локации, куда ведёт заметная доля выборов; для города это обычно он сам
        is_core = self._prob >= core_min_prob
        self.core = pd.Series(i_loc[is_core]).groupby(s_loc[is_core]).agg(set).to_dict()
        # центр локации поиска — медиана координат выбранных из неё объявлений
        self.center = pairs.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
        return self

    def prob(self, search_loc: np.ndarray, item_loc: np.ndarray) -> np.ndarray:
        pos = self._keys.get_indexer(search_loc * LOC_BASE + item_loc)
        return np.where(pos >= 0, self._prob[pos], 0.0)

    def distance_km(self, search_loc: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        c = self.center.reindex(search_loc).to_numpy()
        return haversine_km(c[:, 0], c[:, 1], lat, lon)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def microcat_docs(q_lem: pd.Series, f_lem: pd.Series) -> pd.Series:
    """Документ для классификатора: леммы запроса и леммы фильтров."""
    return q_lem + " | " + f_lem


class MicrocatClassifier:
    """Вероятность подкатегории объявления по тексту запроса и фильтрам.

    Одинаковые пары «документ запроса — подкатегория» схлопываем в одну строку
    с весом: учится в разы быстрее, а качество то же, что на всех парах.
    """

    def __init__(self, seed: int = 0):
        self.word = TfidfVectorizer(
            analyzer="word", token_pattern=r"[^ |]+", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300_000
        )
        self.model = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=15, tol=None, random_state=seed)

    def _vectorize(self, docs, fit=False):
        if fit:
            return sp.hstack([self.word.fit_transform(docs), self.char.fit_transform(docs)]).tocsr()
        return sp.hstack([self.word.transform(docs), self.char.transform(docs)]).tocsr()

    def fit(self, docs: pd.Series, microcat: pd.Series) -> "MicrocatClassifier":
        pairs = pd.DataFrame({"doc": docs.to_numpy(), "y": microcat.to_numpy()})
        grouped = pairs.value_counts().reset_index()
        x = self._vectorize(grouped["doc"], fit=True)
        self.model.fit(x, grouped["y"], sample_weight=grouped["count"])
        self.class_pos = pd.Index(self.model.classes_)
        return self

    def predict_proba(self, docs) -> np.ndarray:
        return self.model.predict_proba(self._vectorize(docs)).astype(np.float32)

    def pair_prob(self, proba: np.ndarray, query_idx: np.ndarray, microcat: np.ndarray) -> np.ndarray:
        cls = self.class_pos.get_indexer(microcat)
        return np.where(cls >= 0, proba[query_idx, np.maximum(cls, 0)], 0.0)


class TextHistory:
    """Что выбирали в train по точно такому же тексту запроса."""

    def fit(self, pairs: pd.DataFrame) -> "TextHistory":
        self.text_count = pairs["text"].value_counts()
        text_code = self.text_count.index.get_indexer(pairs["text"]).astype(np.int64)
        cnt = pd.Series(text_code * MICROCAT_BASE + pairs["item_microcat_id"].to_numpy()).value_counts()
        self._keys = pd.Index(cnt.index)
        self._share = cnt.to_numpy() / self.text_count.to_numpy()[cnt.index.to_numpy() // MICROCAT_BASE]
        return self

    def count(self, texts: pd.Series) -> np.ndarray:
        return self.text_count.reindex(texts).fillna(0).to_numpy()

    def share(self, texts: pd.Series, query_idx: np.ndarray, microcat: np.ndarray) -> np.ndarray:
        """Доля выборов подкатегории по тексту запроса; NaN, если текст не встречался."""
        text_code = self.text_count.index.get_indexer(texts)[query_idx]
        pos = self._keys.get_indexer(text_code.astype(np.int64) * MICROCAT_BASE + microcat)
        share = np.where(pos >= 0, self._share[pos], 0.0)
        return np.where(text_code >= 0, share, np.nan)
