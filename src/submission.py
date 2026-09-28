"""Запись answer.csv и проверка формата перед отправкой."""
import re
from pathlib import Path

import pandas as pd

ITEM_ID = re.compile(r"[0-9a-f]{16}")
QUERY_ID = re.compile(r"[0-9A-Za-z]{16}")


def write_answer(path: Path, query_ids: list[str], predictions: list[list[str]]) -> None:
    answer = pd.DataFrame({"query_id": query_ids, "answer": [" ".join(p) for p in predictions]})
    answer.to_csv(path, index=False)


def validate_answer(
    path: Path, bench_queries: pd.DataFrame, bench_items: pd.DataFrame, k: int = 50
) -> None:
    """Все требования к файлу из условия. Неверный item_id проверка платформы
    пропустит молча, поэтому сверяем каждый id с корпусом."""
    problems = []
    with open(path, "rb") as fh:
        if fh.readline() != b"query_id,answer\n":
            problems.append("первая строка должна быть ровно 'query_id,answer' без BOM и индекса")

    # читаем строками, иначе pandas может превратить id вроде 1e5... в число
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["query_id", "answer"]:
        problems.append(f"колонки {list(df.columns)}")
    if df["query_id"].duplicated().any():
        problems.append("повторяются query_id")
    expected = set(bench_queries["query_id"])
    if set(df["query_id"]) != expected or len(df) != len(expected):
        problems.append("набор query_id не совпадает с benchmark_queries")
    if not df["query_id"].map(lambda s: QUERY_ID.fullmatch(s) is not None).all():
        problems.append("query_id не из 16 символов")

    corpus = set(bench_items["item_id"])
    for qid, answer in zip(df["query_id"], df["answer"]):
        ids = answer.split(" ")
        if len(ids) > k:
            problems.append(f"{qid}: больше {k} объявлений")
        if len(set(ids)) != len(ids):
            problems.append(f"{qid}: повторы внутри ответа")
        if not all(ITEM_ID.fullmatch(i) for i in ids):
            problems.append(f"{qid}: item_id не в формате 16 символов 0-9a-f или лишние пробелы")
        if not corpus.issuperset(ids):
            problems.append(f"{qid}: item_id, которых нет в корпусе")
        if len(problems) > 20:
            break
    if problems:
        raise ValueError("answer.csv не прошёл проверку:\n" + "\n".join(problems))
