import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from dataclasses import dataclass, asdict, replace
from typing import Optional

from src.base_config import _check, TrainConfig, EMBEDDING_SPECS

@dataclass
class BiLSTMConfig(TrainConfig):
    # словник і послідовності
    vocab_max_size: int
    vocab_min_freq: int
    max_length: int
    # ембединги
    embedding: Optional[str]     # None (випадкові) | "glove" | "fasttext"
    embedding_dim: int           # використовується лише коли embedding=None
    freeze_embeddings: bool      # ігнорується коли embedding=None
    # архітектура
    hidden_size: int
    num_layers: int
    dropout: float               # перед лінійним шаром; між шарами LSTM - якщо num_layers > 1

    def __post_init__(self):
        super().__post_init__()
        _check("embedding", self.embedding, {None, "glove", "fasttext"})


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, embedding_dim=128, hidden_size=128, num_layers=1,
                 dropout=0.3, pretrained_weights=None, freeze_embeddings=False):
        super().__init__()

        if pretrained_weights is not None:
            embedding_dim = pretrained_weights.shape[1]
            # clone() щоб тренування finetuned-моделі не змінювало спільну матрицю
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights.clone(), freeze=freeze_embeddings, padding_idx=pad_idx,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        # hidden: [num_layers * 2, B, H]; останні два елементи - forward/backward останнього шару
        representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        representation = self.dropout(representation)
        return self.fc(representation).squeeze(1)

def make_bilstm(cfg, vocab, emb_matrix=None):
    if cfg.embedding is not None and emb_matrix is None:
        # Для відновлення моделі під час тестування: реальні ваги підвантажаться з чекпоінта
        emb_matrix = torch.zeros(len(vocab), EMBEDDING_SPECS[cfg.embedding]["dim"])
    return BiLSTMClassifier(
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        embedding_dim=cfg.embedding_dim,         # ігнорується, якщо є pretrained
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pretrained_weights=emb_matrix,
        freeze_embeddings=cfg.freeze_embeddings,
    )

def bilstm_forward(model, batch, device):
    return model(batch["input_ids"].to(device), batch["lengths"])