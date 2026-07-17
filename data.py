"""Data loading and batching.

Pipeline: read the raw text -> tokenize the whole thing once into a 1-D tensor of
ids -> split into train / validation -> hand out random batches for training.

``get_batch`` is the piece that feeds the model: it grabs random chunks of the
data as inputs (``x``) together with the "next character" targets (``y``), which
are the same chunks shifted one position to the right.
"""

from __future__ import annotations

import torch

import config
from tokenizer import CharTokenizer


def load_text(path: str = config.DATA_PATH) -> str:
    """Read the entire corpus into a single string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_dataset(text: str | None = None):
    """Tokenize the corpus and split it into train / validation tensors.

    Returns ``(tokenizer, train_data, val_data)`` where the two data tensors are
    1-D ``LongTensor``s of character ids. If ``text`` is not given, the corpus is
    read from ``config.DATA_PATH``.
    """
    if text is None:
        text = load_text()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(config.TRAIN_SPLIT * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return tokenizer, train_data, val_data


def get_batch(split_data: torch.Tensor,
              block_size: int = config.BLOCK_SIZE,
              batch_size: int = config.BATCH_SIZE,
              device: str = config.DEVICE):
    """Return one random batch of ``(inputs, targets)`` from ``split_data``.

    * ``x[i]`` is a chunk of ``block_size`` consecutive character ids.
    * ``y[i]`` is that same chunk shifted one step to the right -- i.e. for every
      position, ``y`` holds the character that comes *next*. That is exactly what
      the model learns to predict.

    Both tensors have shape ``(batch_size, block_size)`` and are moved onto
    ``device``.
    """
    # Random start indices. We stop ``block_size`` short of the end so there is
    # always room to take the +1 "next character" for the target.
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
