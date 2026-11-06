"""Єдина точка входу для inference у дашборді (і в скриптах).

    ok, reason = check_inference(run_dir, kind, hf_repo)      # чи можна запускати цей запуск
    predictor  = load_predictor(run_dir, kind, hf_repo, token)
    predictor.predict("текст")  ->  {"prediction": 0|1, "probability": P(AI)}

`kind` — тип моделі з дашборду: bilstm | bilstm_glove | bilstm_fasttext | logreg | distilbert.

Ваги (best_model.pt) беруться з папки запуску; якщо їх там немає, а задано `hf_repo`, завантажуються
з Hugging Face Hub за шляхом  <сценарій>/<run>/best_model.pt  (це зручно для великих файлів DistilBERT).
torch / transformers імпортуються лише тоді, коли справді потрібні.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BILSTM_KINDS = ("bilstm", "bilstm_glove", "bilstm_fasttext")
CHECKPOINT = "best_model.pt"


@dataclass
class Predictor:
    kind: str
    predict: Callable[[str], dict]
    warnings: list[str] = field(default_factory=list)


def _scenario_of(run_dir: Path) -> str:
    return run_dir.parent.parent.name          # runs/<сценарій>/training_logs/<run>


def resolve_checkpoint(run_dir, hf_repo: str | None = None, hf_token: str | None = None) -> Path:
    run_dir = Path(run_dir)
    local = run_dir / CHECKPOINT
    if local.exists():
        return local
    if hf_repo:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(
            repo_id=hf_repo, filename=f"{_scenario_of(run_dir)}/{run_dir.name}/{CHECKPOINT}",
            token=hf_token or None,
        ))
    raise FileNotFoundError(f"Немає {CHECKPOINT} у {run_dir} і не задано HF_MODEL_REPO.")


def check_inference(run_dir, kind: str, hf_repo: str | None = None) -> tuple[bool, str]:
    """(можна_запускати, пояснення). Нічого не завантажує й не імпортує torch."""
    run_dir = Path(run_dir)

    if kind == "logreg":
        if (run_dir / "model.joblib").exists():
            return True, ""
        return False, "немає model.joblib (перезапустіть комірку LogReg з оновленим кодом і зробіть sync)"

    if kind in BILSTM_KINDS or kind == "distilbert":
        if not (run_dir / "config.json").exists():
            return False, "немає config.json"
        if kind in BILSTM_KINDS and not (run_dir / "vocab.json").exists():
            return False, "немає vocab.json (словник не потрапив у репозиторій)"
        if (run_dir / CHECKPOINT).exists():
            return True, ""
        if hf_repo:
            return True, f"ваги завантажаться з Hugging Face Hub ({hf_repo})"
        return False, f"немає {CHECKPOINT} у репозиторії (завеликий файл?) і не задано HF_MODEL_REPO"

    return False, f"inference для типу «{kind}» не підтримується"


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_predictor(run_dir, kind: str, hf_repo: str | None = None, hf_token: str | None = None) -> Predictor:
    run_dir = Path(run_dir)

    if kind == "logreg":
        from src.inference.logreg_inference import load_logreg_for_inference, logreg_version_warning, predict_logreg

        model = load_logreg_for_inference(run_dir)
        warning = logreg_version_warning(run_dir)
        return Predictor(kind, lambda text: predict_logreg(text, model), [warning] if warning else [])

    if kind in BILSTM_KINDS:
        from src.inference.bilstm_inference import load_bilstm_for_inference, predict_bilstm

        device = _device()
        checkpoint = resolve_checkpoint(run_dir, hf_repo, hf_token)
        model, vocab, cfg = load_bilstm_for_inference(run_dir, device, checkpoint)
        return Predictor(kind, lambda text: predict_bilstm(text, model, vocab, cfg, device))

    if kind == "distilbert":
        from src.inference.transformer_inference import load_transformer_for_inference, predict_transformer

        device = _device()
        checkpoint = resolve_checkpoint(run_dir, hf_repo, hf_token)
        model, tokenizer, cfg = load_transformer_for_inference(run_dir, device, checkpoint)
        return Predictor(kind, lambda text: predict_transformer(text, model, tokenizer, cfg, device))

    raise ValueError(f"inference для типу «{kind}» не підтримується")
