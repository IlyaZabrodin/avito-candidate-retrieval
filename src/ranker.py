"""Модель отбора кандидатов и метрика."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def pool_labels(pool: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """1, если кандидат — одно из выбранных по запросу объявлений."""
    relevant = set(
        zip(
            np.repeat(np.arange(len(queries)), queries["relevant"].str.len()),
            queries["relevant"].explode().to_numpy(),
        )
    )
    item_ids = items["item_id"].to_numpy()[pool["item_idx"].to_numpy()]
    return np.fromiter(
        ((q, i) in relevant for q, i in zip(pool["query_idx"].to_numpy(), item_ids)),
        dtype=np.int8,
        count=len(pool),
    )


class Ranker:
    """Бинарный бустинг «выберут / не выберут».

    Порядок внутри топ-50 метрике не важен, поэтому хватает поточечной модели:
    берём 50 кандидатов с наибольшей вероятностью. Негативов в пуле в сотни раз
    больше позитивов, и часть из них выкидываем: это сдвигает вероятности,
    но не порядок кандидатов.
    """

    def __init__(self, neg_frac: float = 0.3, seed: int = 0):
        self.neg_frac, self.seed = neg_frac, seed
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=1000,
            max_leaf_nodes=63,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=seed,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Ranker":
        rng = np.random.default_rng(self.seed)
        keep = (y == 1) | (rng.random(len(y)) < self.neg_frac)
        self.columns = list(X.columns)
        self.model.fit(X.loc[keep].to_numpy(np.float32), y[keep])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.columns].to_numpy(np.float32))[:, 1]


def select_top(pool: pd.DataFrame, scores: np.ndarray, k: int = 50) -> pd.Series:
    """Топ-k позиций корпуса для каждого запроса; при равных скорах — по позиции."""
    frame = pd.DataFrame(
        {"q": pool["query_idx"].to_numpy(), "i": pool["item_idx"].to_numpy(), "s": -scores}
    ).sort_values(["q", "s", "i"], kind="stable")
    return frame.groupby("q", sort=True)["i"].apply(lambda s: s.to_numpy()[:k])


def recall_at_k(predicted: list, relevant: list) -> float:
    """Метрика задачи: доля найденных выбранных объявлений, усреднённая по запросам."""
    return float(np.mean([len(set(p) & set(r)) / len(r) for p, r in zip(predicted, relevant)]))
