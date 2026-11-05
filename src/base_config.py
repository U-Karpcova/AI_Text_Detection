from dataclasses import dataclass, asdict, replace

METRIC_NAMES = {"loss", "accuracy", "precision_ai", "recall_ai", "f1_ai", "macro_f1", "roc_auc"}

def _check(field_name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{field_name}={value!r}: допустимі значення {sorted(allowed, key=str)}")


EMBEDDING_SPECS = {
    "glove": dict(
        url="https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip",
        zip_name="glove.42B.zip",
        member="glove.42B.300d",
        dim=300,
    ),
    "fasttext": dict(
        url="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip",
        zip_name="wiki-news-300d-1M-subword.vec.zip",
        member="wiki-news-300d-1M-subword.vec",
        dim=300,
    ),
}


@dataclass
class TrainConfig:
    """Параметри навчання (BiLSTM і DistilBERT)"""
    optimizer: str               # "adam" (L2 на всі ваги) | "adamw" (без decay для bias/LayerNorm)
    lr: float
    weight_decay: float
    batch_size: int
    eval_batch_size: int
    grad_accum_steps: int        # ефективний batch = batch_size * grad_accum_steps
    grad_clip: float             # 0 = вимкнено
    n_epochs: int
    evals_per_epoch: int         # скільки разів за епоху рахувати val-метрики
    patience: int                # early stopping: лічильник оцінювань
    monitor: str                 # за якою val-метрикою вибирати чекпоінт і зупинятись
    scheduler: str               # "none" | "linear" (warmup + лінійне згасання до 0)
    warmup_ratio: float          # частка кроків на warmup (для scheduler="linear")
    amp: str                     # "none" | "fp16" | "bf16" | "auto" (bf16 на Ampere+, інакше fp16)
    seed: int

    def __post_init__(self):
        _check("optimizer", self.optimizer, {"adam", "adamw"})
        _check("scheduler", self.scheduler, {"none", "linear"})
        _check("amp", self.amp, {"none", "fp16", "bf16", "auto"})
        _check("monitor", self.monitor, METRIC_NAMES)
        if self.grad_accum_steps < 1 or self.evals_per_epoch < 1:
            raise ValueError("grad_accum_steps і evals_per_epoch мають бути ≥ 1")

