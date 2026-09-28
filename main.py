"""Кандидатогенерация для поиска услуг Авито.

    python main.py

Читает data/*.parquet, строит валидацию из train, обучает модели, печатает
Recall@50 на отложенных запросах и пишет answer.csv для бенчмарка.
Промежуточные результаты кэшируются в cache/; удалите папку, чтобы пересчитать всё с нуля.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR, DATA_FILES, dev_split, load_benchmark, load_train
from src.embeddings import DOC_LEN, Encoder, finetune, item_texts, query_texts
from src.features import feature_columns
from src.pipeline import Corpus, Models, candidate_features
from src.ranker import Ranker, pool_labels, recall_at_k, select_top
from src.stats import LocationStats, MicrocatClassifier, TextHistory, microcat_docs
from src.submission import validate_answer, write_answer
from src.text import Lemmatizer, item_fields

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
ANSWER = ROOT / "answer.csv"

SEED = 0
N_VAL, N_RANK = 5_000, 20_000  # отложенные запросы: оценка качества и обучение модели отбора
N_FINETUNE_PAIRS = 100_000
TOP_K = 50

START = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - START) / 60:5.1f} мин] {msg}", flush=True)


def cached_fields(name: str, items: pd.DataFrame, lemmatizer: Lemmatizer) -> dict[str, list[str]]:
    path = CACHE / f"fields_{name}.parquet"
    if not path.exists():
        pd.DataFrame(item_fields(items, lemmatizer)).to_parquet(path)
    return pd.read_parquet(path).to_dict("list")


def cached_embeddings(name: str, items: pd.DataFrame, encoder: Encoder) -> np.ndarray:
    # рядом с весами энкодера: новые веса — новые эмбеддинги
    path = CACHE / "encoder" / f"emb_{name}.npy"
    if not path.exists():
        np.save(path, encoder.encode(item_texts(items), DOC_LEN))
    return np.load(path)


def train_encoder(fit: pd.DataFrame) -> Encoder:
    path = CACHE / "encoder"
    if not path.exists():
        pairs = (
            fit[["search_query", "item_id", "item_title_raw", "item_infm_params_text"]]
            .drop_duplicates(["search_query", "item_id"])
            .sample(frac=1, random_state=SEED)
            .head(N_FINETUNE_PAIRS)
        )
        # пишем во временную папку: прерванный запуск не оставит в кэше недописанную модель
        tmp = CACHE / "encoder_tmp"
        finetune(query_texts(pairs), item_texts(pairs), pairs["item_id"].to_numpy(), tmp, seed=SEED)
        tmp.rename(path)
    return Encoder(path)


def predict_top(
    features: pd.DataFrame, n_queries: int, ranker: Ranker, items: pd.DataFrame
) -> list[list[str]]:
    """50 лучших item_id на каждый запрос по скору модели отбора."""
    top = select_top(features, ranker.predict(features), TOP_K)
    ids = items["item_id"].to_numpy()
    return [list(ids[top[i]]) for i in range(n_queries)]


def main() -> None:
    missing = [name for name in DATA_FILES if not (DATA_DIR / name).exists()]
    if missing:
        raise SystemExit(f"В {DATA_DIR} нет файлов: {', '.join(missing)}")
    CACHE.mkdir(exist_ok=True)

    # данные и разбиение train: fit для статистик, rank для модели отбора, val для оценки
    bench_queries, bench_items = load_benchmark()
    train_queries, train_items = load_train()
    split = dev_split(train_queries, train_items, bench_queries, len(bench_items), N_VAL, N_RANK, seed=SEED)
    fit = split["fit"].merge(train_items, on="item_id")
    log(f"данные: {len(fit)} пар для статистик, {N_RANK} запросов для модели отбора, {N_VAL} для оценки")

    # всё, что учится на fit
    lemmatizer = Lemmatizer()
    fit_docs = microcat_docs(
        pd.Series(lemmatizer(fit["text"])), pd.Series(lemmatizer(fit["search_infm_params_text"]))
    )
    models = Models(
        lemmatizer=lemmatizer,
        loc_stats=LocationStats().fit(fit),
        history=TextHistory().fit(fit),
        microcat=MicrocatClassifier(seed=SEED).fit(fit_docs, fit["item_microcat_id"]),
        encoder=train_encoder(fit),
    )
    log("статистики, классификатор подкатегорий и энкодер готовы")

    # модель отбора учится на кандидатах отложенных запросов из корпуса, устроенного как бенчмарк
    dev = Corpus(
        split["corpus"],
        cached_fields("dev", split["corpus"], lemmatizer),
        cached_embeddings("dev", split["corpus"], models.encoder),
        models,
    )
    rank_features, rank_queries = candidate_features(split["rank"], dev, models)
    ranker = Ranker(seed=SEED).fit(
        rank_features[feature_columns(rank_features)], pool_labels(rank_features, rank_queries, dev.items)
    )
    log(f"модель отбора обучена: {len(rank_features)} пар, {len(ranker.columns)} признаков")

    # оценка на запросах, которые не участвовали ни в каком обучении
    val_features, val_queries = candidate_features(split["val"], dev, models)
    relevant = val_queries["relevant"].tolist()
    pool = val_features.groupby("query_idx")["item_idx"].agg(list)
    pool_recall = recall_at_k([dev.items["item_id"].to_numpy()[i] for i in pool], relevant)
    val_recall = recall_at_k(predict_top(val_features, len(val_queries), ranker, dev.items), relevant)
    log(f"валидация: полнота пула {pool_recall:.4f}, Recall@{TOP_K} {val_recall:.4f}")

    # те же модели без изменений — на бенчмарк
    bench = Corpus(
        bench_items,
        cached_fields("bench", bench_items, lemmatizer),
        cached_embeddings("bench", bench_items, models.encoder),
        models,
    )
    bench_features, bench_queries = candidate_features(bench_queries, bench, models)
    predictions = predict_top(bench_features, len(bench_queries), ranker, bench_items)
    write_answer(ANSWER, bench_queries["query_id"].tolist(), predictions)
    validate_answer(ANSWER, bench_queries, bench_items, TOP_K)
    log(f"answer.csv записан и проверен: {len(predictions)} запросов по {TOP_K} кандидатов")


if __name__ == "__main__":
    main()
