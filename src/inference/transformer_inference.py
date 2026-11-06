"""Inference для DistilBERT (model_type="distilbert").

Архітектура відновлюється так само, як у build_transformer(cfg, pretrained=False) з ноутбука: лише
конфігурація, ваги беруться з best_model.pt. Токенізатор і конфіг завантажуються з Hugging Face Hub
за cfg["model_name"] (потрібен інтернет під час першого запуску; далі лежать у кеші).
"""
import json
from pathlib import Path


def load_transformer_for_inference(run_dir, device, checkpoint_path=None):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if meta["model_type"] != "distilbert":
        raise ValueError(f"{run_dir.name} is not a DistilBERT run: {meta['model_type']}")
    cfg = meta["config"]

    hf_config = AutoConfig.from_pretrained(
        cfg["model_name"],
        num_labels=1,
        dropout=cfg["dropout"],
        attention_dropout=cfg["attention_dropout"],
        seq_classif_dropout=cfg["classifier_dropout"],
    )
    model = AutoModelForSequenceClassification.from_config(hf_config)

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else run_dir / "best_model.pt"
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    return model, tokenizer, cfg


def predict_transformer(text, model, tokenizer, cfg, device):
    import torch

    # як і під час навчання: беремо початок тексту, обрізаємо до max_length токенів
    encoded = tokenizer(str(text), truncation=True, max_length=cfg["max_length"], return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}

    with torch.no_grad():
        logits = model(**encoded).logits.view(-1)
        probability = torch.sigmoid(logits).item()

    return {"prediction": int(probability >= 0.5), "probability": probability}
