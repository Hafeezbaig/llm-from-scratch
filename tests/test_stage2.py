"""Stage 2 tests: the bigram model's shapes, loss, training and generation.

Runs two ways -- no test framework required:

    python tests/test_stage2.py     # plain runner
    pytest tests/                   # if you have pytest installed

Like Stage 1, the tests use a small synthetic string and run on the CPU, so they
are fast and do not depend on the downloaded dataset. To train on the real data
and see generated text, run ``python bigram.py``.
"""

from __future__ import annotations

import math
import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import config  # noqa: E402
from bigram import BigramLanguageModel  # noqa: E402
from data import build_dataset, get_batch  # noqa: E402
from train import train  # noqa: E402

SAMPLE = "Hello, Watson! The game is afoot.\n" * 100


def _setup():
    """A fresh, seeded, untrained model plus the sample data split."""
    torch.manual_seed(config.SEED)
    tok, train_data, val_data = build_dataset(SAMPLE)
    model = BigramLanguageModel(tok.vocab_size)
    return tok, model, train_data, val_data


def test_forward_shapes():
    """Logits are (batch, time, vocab); the loss is a single number."""
    tok, model, train_data, _ = _setup()
    x, y = get_batch(train_data, block_size=8, batch_size=4, device="cpu")
    logits, loss = model(x, y)
    assert logits.shape == (4, 8, tok.vocab_size)
    assert loss.shape == ()


def test_no_targets_means_no_loss():
    """While generating there is no answer to compare against, so no loss."""
    _, model, train_data, _ = _setup()
    x, _ = get_batch(train_data, block_size=8, batch_size=4, device="cpu")
    _, loss = model(x)
    assert loss is None


def test_untrained_loss_is_about_random_guessing():
    """Before training, the loss should sit near ln(vocab_size), a blind guess.

    The table starts as small random numbers, so it lands a bit above the
    perfectly even guess. Far off (say 10+) would mean something is broken.
    """
    tok, model, train_data, _ = _setup()
    x, y = get_batch(train_data, device="cpu")
    _, loss = model(x, y)
    even_guess = math.log(tok.vocab_size)
    assert even_guess - 0.5 < loss.item() < even_guess + 1.5


def test_training_lowers_loss():
    """A short training run must push the loss well below where it started."""
    _, model, train_data, val_data = _setup()
    history = train(model, train_data, val_data, max_iters=300,
                    eval_interval=100, eval_iters=10,
                    learning_rate=config.BIGRAM_LEARNING_RATE, device="cpu",
                    verbose=False)
    first_train, last_train = history[0][1], history[-1][1]
    assert last_train < first_train - 1.0


def test_generate_length_and_valid_ids():
    """generate() keeps the prompt and adds exactly max_new_tokens valid ids."""
    tok, model, _, _ = _setup()
    start = torch.zeros((2, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=20)
    assert out.shape == (2, 21)
    assert torch.equal(out[:, :1], start)
    assert ((out >= 0) & (out < tok.vocab_size)).all()


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


if __name__ == "__main__":
    n_failed = _run_all()
    print(f"\n{'ALL TESTS PASSED' if n_failed == 0 else f'{n_failed} TEST(S) FAILED'}")
    sys.exit(1 if n_failed else 0)
