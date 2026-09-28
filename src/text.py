"""Нормализация текста: токены в нижнем регистре и леммы."""
import re
from collections.abc import Iterable

import pymorphy3

TOKEN_RE = re.compile(r"[a-zа-я0-9]+")


class Lemmatizer:
    """pymorphy3 с кэшем: словарь уникальных слов намного меньше числа токенов."""

    def __init__(self):
        self._morph = pymorphy3.MorphAnalyzer()
        self._cache: dict[str, str] = {}

    def lemma(self, token: str) -> str:
        lemma = self._cache.get(token)
        if lemma is None:
            if token.isalpha():
                lemma = self._morph.parse(token)[0].normal_form.replace("ё", "е")
            else:
                lemma = token
            self._cache[token] = lemma
        return lemma

    def __call__(self, texts: Iterable[str], max_chars: int | None = None) -> list[str]:
        out = []
        for text in texts:
            text = text[:max_chars].lower().replace("ё", "е")
            out.append(" ".join(self.lemma(t) for t in TOKEN_RE.findall(text)))
        return out


def item_fields(items, lemmatizer: Lemmatizer) -> dict[str, list[str]]:
    """Лемматизированные поля объявления. Описание обрезаем до 3000 символов:
    у большинства объявлений оно короче (медиана около 1000), а хвост самых
    длинных заметно замедляет расчёт."""
    return {
        "title": lemmatizer(items["item_title_raw"]),
        "params": lemmatizer(items["item_infm_params_text"]),
        "desc": lemmatizer(items["item_description_raw"], max_chars=3000),
    }
