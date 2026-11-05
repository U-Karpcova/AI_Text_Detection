import json
import torch
from src.models.bilstm import BiLSTMConfig, make_bilstm
from src.preprocessing.vocabulary import Vocabulary

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

    # Тут використати ту саму логіку завантаження checkpoint,
    # яку зараз використовує test_model()
    ...

    model.eval()

    return model, vocab, cfg