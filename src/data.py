import torch


def make_batch(batch_size, seq_len, vocab_size, device, generator=None):
    # quick synthetic next-token batch
    tokens = torch.randint(
        0, vocab_size, (batch_size, seq_len + 1), device=device, generator=generator
    )
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y
