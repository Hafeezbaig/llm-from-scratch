"""Character-level tokenizer.

The simplest tokenizer there is: every unique character in the training text
becomes one integer id. ``encode`` turns text into a list of ids and ``decode``
turns those ids back into text.

We deliberately do NOT use subword / BPE tokenization here. Character-level keeps
the whole idea to a handful of lines and lets us focus on the model itself.
Swapping in a smarter tokenizer is a later upgrade, not a starting point.
"""

from __future__ import annotations


class CharTokenizer:
    """Maps characters <-> integer ids using a fixed vocabulary.

    The vocabulary is the sorted set of unique characters found in the text it is
    built from. Sorting makes the mapping deterministic: the same text always
    produces the same ids. A character that was not present at build time cannot
    be encoded (it would raise a ``KeyError``) -- expected for a toy tokenizer.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.chars = chars
        self.vocab_size = len(chars)
        self._stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
        self._itos = {i: ch for i, ch in enumerate(chars)}  # int -> string

    def encode(self, text: str) -> list[int]:
        """Turn a string into a list of integer ids."""
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Turn a list (or any iterable) of integer ids back into a string."""
        return "".join(self._itos[i] for i in ids)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
