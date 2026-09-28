"""Лексический индекс корпуса: BM25 и доля слов запроса по каждому полю."""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bm25_weights(tf: sp.csr_matrix, k1: float = 1.2, b: float = 0.75) -> tuple[sp.csr_matrix, np.ndarray]:
    """Матрица весов BM25 (слова x документы) и idf слов."""
    tf = tf.tocsr().astype(np.float32)
    n_docs = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    doc_len = np.asarray(tf.sum(axis=1)).ravel()
    norm = k1 * (1 - b + b * doc_len / max(doc_len.mean(), 1e-9))
    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows])
    return (tf @ sp.diags(idf)).T.tocsr(), idf


class TextIndex:
    """Поля объявления в общем словаре лемм.

    Для каждого поля храним веса BM25. По ним же считается, какая доля слов
    запроса (с весом idf по всему объявлению) нашлась в поле.
    """

    def __init__(self, fields: dict[str, list[str]]):
        self.vectorizer = CountVectorizer(analyzer=str.split, dtype=np.float32)
        self.vectorizer.fit(doc for docs in fields.values() for doc in docs)
        tf = {name: self.vectorizer.transform(docs) for name, docs in fields.items()}
        tf["all"] = sum(tf.values())
        self.bm25 = {}
        for name, m in tf.items():
            self.bm25[name], idf = bm25_weights(m)
            self.bm25[name].sort_indices()  # pair_scores ищет в строках бинарным поиском
            if name == "all":
                self.idf = idf

    def query_matrix(self, queries: list[str]) -> sp.csr_matrix:
        q = self.vectorizer.transform(queries)
        q.data[:] = 1.0  # повтор слова в коротком запросе не должен удваивать вес
        return q.tocsr()

    def scores(self, q: sp.csr_matrix, field: str) -> sp.csr_matrix:
        return (q @ self.bm25[field]).tocsr()

    def pair_scores(
        self, q: sp.csr_matrix, query_idx: np.ndarray, item_idx: np.ndarray, field: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """BM25 и доля idf слов запроса, найденных в поле, для пар (запрос, документ).

        Пары сгруппированы по запросу. Для каждого слова запроса ищем кандидатов
        в строке матрицы весов бинарным поиском: слов в запросе мало, так дешевле,
        чем считать скоры по всему корпусу.
        """
        w = self.bm25[field]
        bm25 = np.zeros(len(query_idx), dtype=np.float32)
        matched = np.zeros(len(query_idx), dtype=np.float32)
        bounds = np.flatnonzero(np.diff(query_idx)) + 1
        for start, end in zip(np.r_[0, bounds], np.r_[bounds, len(query_idx)]):
            i, cand = query_idx[start], item_idx[start:end]
            for t in q.indices[q.indptr[i] : q.indptr[i + 1]]:
                docs = w.indices[w.indptr[t] : w.indptr[t + 1]]
                if len(docs) == 0:
                    continue
                pos = np.minimum(np.searchsorted(docs, cand), len(docs) - 1)
                hit = docs[pos] == cand
                bm25[start:end] += np.where(hit, w.data[w.indptr[t] + pos], 0.0)
                matched[start:end] += hit * self.idf[t]
        q_idf = np.asarray(q @ self.idf).ravel()[query_idx]
        return bm25, matched / np.maximum(q_idf, 1e-6)


class CharIndex:
    """Косинус по символьным n-граммам: ловит опечатки и слитное/раздельное написание."""

    def __init__(self, docs: list[str]):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32
        )
        self.matrix = self.vectorizer.fit_transform(docs).tocsr()

    def pair_cosine(self, queries: list[str], query_idx: np.ndarray, doc_idx: np.ndarray) -> np.ndarray:
        q = self.vectorizer.transform(queries).tocsr()
        out = np.zeros(len(query_idx), dtype=np.float32)
        # пары сгруппированы по запросу: на каждый запрос одно умножение по его кандидатам
        bounds = np.flatnonzero(np.diff(query_idx)) + 1
        for start, end in zip(np.r_[0, bounds], np.r_[bounds, len(query_idx)]):
            i = query_idx[start]
            out[start:end] = (self.matrix[doc_idx[start:end]] @ q[i].T).toarray().ravel()
        return out
