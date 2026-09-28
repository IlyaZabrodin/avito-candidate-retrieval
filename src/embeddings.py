"""Эмбеддинги e5: дообучение на парах из train и кодирование текстов."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "intfloat/multilingual-e5-small"
BASE_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
QUERY_LEN, DOC_LEN = 32, 48


def item_texts(items) -> list[str]:
    # заголовок и начало параметров, где стоят вид и тип услуги
    return ("passage: " + items["item_title_raw"] + ". " + items["item_infm_params_text"].str[:120]).tolist()


def query_texts(queries) -> list[str]:
    return ("query: " + queries["search_query"]).tolist()


def inference_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mean_pool(model, batch) -> torch.Tensor:
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9), dim=-1)


def finetune(
    queries: list[str],
    docs: list[str],
    item_ids: np.ndarray,
    out_dir: Path,
    batch_size: int = 128,
    lr: float = 3e-5,
    seed: int = 0,
) -> None:
    """Контрастивное дообучение: для запроса правильный ответ — его объявление,
    остальные объявления батча — негативы.

    Учим на CPU: на MPS обратный проход недетерминирован, и веса от запуска
    к запуску немного расходятся, а ответ должен воспроизводиться.
    """
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = AutoModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION).train()
    steps = len(queries) // batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # линейный разогрев на первых 5% шагов, дальше линейное затухание до нуля
    warmup = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * (steps - s) / steps
    )

    for step in range(steps):
        sl = slice(step * batch_size, (step + 1) * batch_size)
        q = tokenizer(queries[sl], padding=True, truncation=True, max_length=QUERY_LEN, return_tensors="pt")
        d = tokenizer(docs[sl], padding=True, truncation=True, max_length=DOC_LEN, return_tensors="pt")
        logits = mean_pool(model, q) @ mean_pool(model, d).T * 20.0  # температура 0.05, как в e5
        # одно и то же объявление дважды в батче — не негатив для самого себя
        ids = item_ids[sl]
        dup = torch.from_numpy((ids[:, None] == ids[None, :]) & ~np.eye(len(ids), dtype=bool))
        loss = F.cross_entropy(logits.masked_fill(dup, -1e4), torch.arange(len(ids)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"  дообучение энкодера: шаг {step}/{steps}, loss {loss.item():.3f}", flush=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


class Encoder:
    """Кодирует тексты дообученной моделью: на GPU или MPS, если они есть, иначе на CPU."""

    def __init__(self, path: str | Path, batch_size: int = 256):
        self.device = inference_device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode(self, texts: list[str], max_length: int) -> np.ndarray:
        # батчи из текстов близкой длины — меньше паддинга
        order = np.argsort([-len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = self.tokenizer(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            out[idx] = mean_pool(self.model, batch).float().cpu().numpy()
        return out
