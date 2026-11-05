import json
import torch
from pathlib import Path

from src.models.bilstm import BiLSTMConfig, make_bilstm
from src.preprocessing.vocabulary import Vocabulary
from src.preprocessing.text_encoder import encode_texts
from src.paths import RUNS_DIR


def load_bilstm_for_inference(run_dir, device):
    run_dir = Path(run_dir)
    meta = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )

    if meta["model_type"] != "bilstm":
        raise ValueError(
            f"{run_dir.name} is not a BiLSTM run: "
            f"{meta['model_type']}"
        )

    cfg = BiLSTMConfig(**meta["config"])

    vocab = Vocabulary.from_itos(
        json.loads(
            (run_dir / "vocab.json").read_text(encoding="utf-8")
        ),
        cfg.vocab_min_freq,
        cfg.vocab_max_size,
    )

    model = make_bilstm(cfg, vocab).to(device)
    checkpoint_path = run_dir / "best_model.pt"

    # завантажити найкращі ваги
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()

    return model, vocab, cfg


def predict_bilstm(text, model, vocab, cfg, device):
    sequence = encode_texts(
        [text],
        vocab,
        max_length=cfg.max_length,
    )[0]

    # Batch з одного прикладу
    input_ids = torch.tensor(
        [sequence],
        dtype=torch.long,
        device=device,
    )

    lengths = torch.tensor(
        [len(sequence)],
        dtype=torch.long,
    )

    # Inference без обчислення градієнтів
    with torch.no_grad():
        logits = model(input_ids, lengths)

        # BCEWithLogitsLoss означає, що модель повертає logits
        probability = torch.sigmoid(logits).item()

    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "probability": probability,
    }


# INFERENCE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vocab, cfg = load_bilstm_for_inference(
        run_dir=RUNS_DIR / "defactify" / "training_logs" / "bilstm_glove_frozen",
        device=DEVICE
    )

from src.inference.sample_text import text

result = predict_bilstm(
        text,
        model,
        vocab,
        cfg,
        DEVICE
    )

print(result)
