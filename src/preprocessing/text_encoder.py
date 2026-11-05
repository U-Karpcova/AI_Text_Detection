def encode_texts(texts, vocab, max_length=512):
    sequences = []
    for text in texts:
        ids = vocab.encode(text, max_length=max_length)
        if len(ids) == 0:                 # не допускаємо послідовностей нульової довжини
            ids = [vocab.unk_idx]
        sequences.append(ids)
    return sequences