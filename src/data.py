import torch


def make_random_batch(batch_size, seq_len, vocab_size, device, generator=None):
    # quick synthetic next-token batch
    tokens = torch.randint(
        0, vocab_size, (batch_size, seq_len + 1), device=device, generator=generator
    )
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y


class TextBatcher:
    # simple text loader for next-token prediction
    def __init__(self, path, seq_len, device, *, use_bytes=True, encoding="utf-8"):
        self.seq_len = seq_len
        self.device = device

        if use_bytes:
            # treat the file as raw bytes (vocab size = 256)
            data = open(path, "rb").read()
            tokens = torch.tensor(list(data), dtype=torch.long)
            self.vocab_size = 256
        else:
            text = open(path, "r", encoding=encoding).read()
            vocab = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(vocab)}
            tokens = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
            self.vocab_size = len(vocab)

        if tokens.numel() <= seq_len + 1:
            raise ValueError("text file is too small for the chosen seq_len")

        self.tokens = tokens

    def get_batch(self, batch_size, generator=None):
        max_start = self.tokens.numel() - self.seq_len - 1
        idx = torch.randint(0, max_start, (batch_size,), generator=generator)
        idx = idx.tolist()

        x = torch.stack([self.tokens[i : i + self.seq_len] for i in idx])
        y = torch.stack([self.tokens[i + 1 : i + self.seq_len + 1] for i in idx])

        return x.to(self.device), y.to(self.device)
