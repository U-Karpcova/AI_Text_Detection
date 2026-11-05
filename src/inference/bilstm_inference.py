import json
import torch
from pathlib import Path
from src.models.bilstm import BiLSTMConfig, make_bilstm
from src.preprocessing.vocabulary import Vocabulary
from src.preprocessing.text_encoder import encode_texts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_bilstm_for_inference(run_dir):
    meta = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )

    assert meta["model_type"] == "bilstm"

    cfg = BiLSTMConfig(**meta["config"])

    vocab = Vocabulary.from_itos(
        json.loads(
            (run_dir / "vocab.json").read_text(encoding="utf-8")
        ),
        cfg.vocab_min_freq,
        cfg.vocab_max_size,
    )

    model = make_bilstm(cfg, vocab).to(DEVICE)
    checkpoint_path = run_dir / "best_model.pt"

    # завантажити найкращі ваги
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()

    return model, vocab, cfg


def predict_bilstm(text, model, vocab, cfg):
    sequence = encode_texts(
        [text],
        vocab,
        cfg.max_length,
    )

    # створюємо batch
    # запускаємо model у torch.no_grad()
    # отримуємо probability
    # повертаємо результат


if __name__ == 'main':
    model, vocab, cfg = load_bilstm_for_inference(
        Path("training_logs/my_bilstm_run")
    )

    result = predict_bilstm(
        "This is some completely new text.",
        model,
        vocab,
        cfg,
    )

    print(result)
