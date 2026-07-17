"""Stage 1 tests: the tokenizer round-trips and batches are shaped/aligned right.

Runs two ways -- no test framework required:

    python tests/test_stage1.py     # plain runner, also prints a live demo
    pytest tests/                   # if you have pytest installed

The tests use a small synthetic string so they do not depend on the downloaded
dataset. The demo at the bottom uses the real data/input.txt if it is present.
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import config  # noqa: E402
from data import build_dataset, get_batch  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

SAMPLE = "Hello, Watson! The game is afoot.\n" * 100


def test_encode_decode_roundtrip():
    """decode(encode(text)) must give back exactly the original text."""
    tok = CharTokenizer(SAMPLE)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_encoded_ids_are_valid():
    """Every id is an int inside [0, vocab_size)."""
    tok = CharTokenizer(SAMPLE)
    ids = tok.encode(SAMPLE)
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_vocab_is_sorted_and_unique():
    """Vocabulary is the sorted set of unique characters -- deterministic."""
    tok = CharTokenizer("banana")
    assert tok.chars == ["a", "b", "n"]
    assert tok.vocab_size == 3


def test_batch_shapes_and_alignment():
    """x and y are (batch, block); y is x shifted one step right."""
    torch.manual_seed(config.SEED)
    _, train_data, _ = build_dataset(SAMPLE)
    x, y = get_batch(train_data, block_size=8, batch_size=4, device="cpu")
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # The target is the "next character", so x[:, 1:] must equal y[:, :-1].
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_train_val_split_sizes():
    """Train/val split matches config.TRAIN_SPLIT and covers all the data."""
    _, train_data, val_data = build_dataset(SAMPLE)
    total = len(train_data) + len(val_data)
    tok = CharTokenizer(SAMPLE)
    assert total == len(tok.encode(SAMPLE))
    assert len(train_data) == int(config.TRAIN_SPLIT * total)


def _run_all() -> int:
    """Tiny test runner so this works without pytest installed."""
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    return failed


def _demo() -> None:
    """Show, on the real dataset, what Stage 1 actually produces."""
    if not os.path.exists(config.DATA_PATH):
        print(f"\n(skipping demo: {config.DATA_PATH} not found -- "
              f"run `python scripts/prepare_data.py` first)")
        return

    print("\n" + "=" * 60)
    print("DEMO on real data:", config.DATA_PATH)
    print("=" * 60)

    torch.manual_seed(config.SEED)
    tok, train_data, val_data = build_dataset()
    print(f"device                : {config.DEVICE}")
    print(f"total characters      : {len(train_data) + len(val_data):,}")
    print(f"vocabulary size       : {tok.vocab_size}")
    print(f"train / val tokens    : {len(train_data):,} / {len(val_data):,}")

    snippet = "Elementary, my dear Watson."
    ids = tok.encode(snippet)
    print(f"\nencode({snippet!r})")
    print(f"  -> {ids}")
    print(f"decode(...) -> {tok.decode(ids)!r}")
    print(f"round-trip ok         : {tok.decode(ids) == snippet}")

    x, y = get_batch(train_data, block_size=8, batch_size=2)
    print(f"\nsample batch  x.shape={tuple(x.shape)}  y.shape={tuple(y.shape)}")
    print("first sequence in the batch (input -> next-char target):")
    for inp, tgt in zip(x[0].tolist(), y[0].tolist()):
        print(f"  {tok.decode([inp])!r:>6}  ->  {tok.decode([tgt])!r}")


if __name__ == "__main__":
    n_failed = _run_all()
    _demo()
    print(f"\n{'ALL TESTS PASSED' if n_failed == 0 else f'{n_failed} TEST(S) FAILED'}")
    sys.exit(1 if n_failed else 0)
