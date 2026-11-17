"""Stage 3 tests: each transformer piece, then the whole GPT.

Runs two ways -- no test framework required:

    python tests/test_stage3.py     # plain runner
    pytest tests/                   # if you have pytest installed

The pieces are tested in the order they are built: one head, many heads,
feed-forward, block, full model. The most important check is that no position
can see the future. Everything runs on a tiny model on the CPU, so it is fast
and does not need the downloaded dataset. To train the real model, run
``python train.py``.
"""

from __future__ import annotations

import math
import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import config  # noqa: E402
from data import build_dataset, get_batch  # noqa: E402
from model import GPT, Block, FeedForward, Head, MultiHeadAttention  # noqa: E402
from train import train  # noqa: E402

SAMPLE = "Hello, Watson! The game is afoot.\n" * 100

# A tiny model: same shape of code, far fewer numbers.
N_EMBD, N_HEAD, N_LAYER, BLOCK = 16, 4, 2, 8


def _x(batch: int = 2, block: int = BLOCK) -> torch.Tensor:
    """A random stand-in for a batch of position vectors."""
    torch.manual_seed(config.SEED)
    return torch.randn(batch, block, N_EMBD)


def test_head_shape():
    """One head turns each 16-number position into a 4-number summary."""
    head = Head(N_EMBD, head_size=4, block_size=BLOCK)
    assert head(_x()).shape == (2, BLOCK, 4)


def test_head_cannot_see_the_future():
    """Changing later positions must not change the output at earlier ones.

    If it did, the model could cheat during training by peeking at the very
    character it is supposed to predict.
    """
    head = Head(N_EMBD, head_size=4, block_size=BLOCK)
    x = _x()
    changed = x.clone()
    changed[:, 5:] = torch.randn(2, BLOCK - 5, N_EMBD)
    before, after = head(x), head(changed)
    assert torch.allclose(before[:, :5], after[:, :5])        # past: untouched
    assert not torch.allclose(before[:, 5:], after[:, 5:])    # future: changed


def test_multi_head_shape():
    """4 heads of 4 numbers each are joined back into 16 numbers."""
    mha = MultiHeadAttention(N_EMBD, N_HEAD, BLOCK)
    assert mha(_x()).shape == (2, BLOCK, N_EMBD)


def test_feed_forward_shape():
    """Feed-forward works on each position alone and keeps its size."""
    assert FeedForward(N_EMBD)(_x()).shape == (2, BLOCK, N_EMBD)


def test_block_shape():
    """A block keeps the shape, so blocks can be stacked on top of each other."""
    assert Block(N_EMBD, N_HEAD, BLOCK)(_x()).shape == (2, BLOCK, N_EMBD)


def test_gpt_shapes_and_untrained_loss():
    """Logits are (batch, time, vocab); untrained loss is near a blind guess."""
    torch.manual_seed(config.SEED)
    tok, train_data, _ = build_dataset(SAMPLE)
    model = GPT(tok.vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK)
    x, y = get_batch(train_data, block_size=BLOCK, batch_size=4, device="cpu")
    logits, loss = model(x, y)
    assert logits.shape == (4, BLOCK, tok.vocab_size)
    even_guess = math.log(tok.vocab_size)
    assert even_guess - 0.5 < loss.item() < even_guess + 1.0


def test_gpt_cannot_see_the_future():
    """The same no-peeking rule holds through the whole stack of blocks."""
    torch.manual_seed(config.SEED)
    model = GPT(vocab_size=20, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER,
                block_size=BLOCK)
    model.eval()  # dropout off (Stage 4), so the two runs are comparable
    idx = torch.randint(20, (1, BLOCK))
    changed = idx.clone()
    changed[0, -1] = (idx[0, -1] + 1) % 20
    before, _ = model(idx)
    after, _ = model(changed)
    assert torch.allclose(before[:, :-1], after[:, :-1])


def test_generate_goes_past_the_window():
    """Generating more characters than block_size works (it keeps the latest)."""
    torch.manual_seed(config.SEED)
    model = GPT(vocab_size=20, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER,
                block_size=BLOCK)
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=3 * BLOCK)
    assert out.shape == (1, 3 * BLOCK + 1)
    assert ((out >= 0) & (out < 20)).all()


def test_training_lowers_loss():
    """A short training run must push the loss well below where it started."""
    torch.manual_seed(config.SEED)
    tok, train_data, val_data = build_dataset(SAMPLE)
    # train() draws batches of config.BLOCK_SIZE, so the window must match.
    model = GPT(tok.vocab_size, N_EMBD, N_HEAD, N_LAYER, config.BLOCK_SIZE)
    history = train(model, train_data, val_data, max_iters=200,
                    eval_interval=100, eval_iters=5, device="cpu",
                    verbose=False)
    first_train, last_train = history[0][1], history[-1][1]
    assert last_train < first_train - 1.0


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
