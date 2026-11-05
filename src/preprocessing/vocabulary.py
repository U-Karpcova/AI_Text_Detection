import re
from collections import Counter

class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, min_freq=2, max_size=30_000):
        self.min_freq = min_freq
        self.max_size = max_size

        self.itos = [self.PAD_TOKEN, self.UNK_TOKEN]
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    @staticmethod
    def tokenize(text):
        """Проста токенізація для англомовного baseline."""
        text = str(text).lower()
        return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text)

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        # Відсікаємо рідкісні слова та спеціальні токени.
        tokens = [
            token for token, count in counter.most_common()
            if count >= self.min_freq and token not in (self.PAD_TOKEN, self.UNK_TOKEN)
        ]

        # max_size стосується звичайних токенів, спеціальні додаються окремо.
        tokens = tokens[:self.max_size]

        self.itos = [self.PAD_TOKEN, self.UNK_TOKEN] + tokens
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        return self

    @classmethod
    def from_itos(cls, itos, min_freq=2, max_size=30_000):
        """Відновлює словник зі збереженого списку токенів (vocab.json у папці запуску)."""
        vocab = cls(min_freq=min_freq, max_size=max_size)
        vocab.itos = list(itos)
        vocab.stoi = {token: idx for idx, token in enumerate(vocab.itos)}
        return vocab

    def encode(self, text, max_length=512):
        tokens = self.tokenize(text)[:max_length]          # обрізаємо задовгу послідовність
        return [self.stoi.get(token, self.stoi[self.UNK_TOKEN]) for token in tokens]

    def decode(self, ids):
        return [self.itos[idx] for idx in ids]

    def __len__(self):
        return len(self.itos)

    @property
    def pad_idx(self):
        return self.stoi[self.PAD_TOKEN]

    @property
    def unk_idx(self):
        return self.stoi[self.UNK_TOKEN]