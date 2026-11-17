"""Stage 4 tests: dropout, the learning-rate schedule, checkpoints and the plot.

Runs two ways -- no test framework required:

    python tests/test_stage4.py     # plain runner
    pytest tests/                   # if you have pytest installed

Everything uses a tiny model on the CPU and a temporary folder, so it is fast
and leaves no files behind. To train the real model, run ``python train.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import config  # noqa: E402
from data import build_dataset  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from train import get_lr, plot_losses, save_checkpoint, train  # noqa: E402

SAMPLE = "Hello, Watson! The game is afoot.\n" * 100


def _tiny_gpt(vocab_size: int = 20, dropout: float = 0.0) -> GPT:
    torch.manual_seed(config.SEED)
    return GPT(vocab_size, n_embd=16, n_head=4, n_layer=2, block_size=8,
               dropout=dropout)


def test_dropout_only_while_training():
    """In train mode dropout makes each run differ; eval mode turns it off."""
    model = _tiny_gpt(dropout=0.5)
    idx = torch.randint(20, (2, 8))
    model.train()
    assert not torch.equal(model(idx)[0], model(idx)[0])
    model.eval()
    assert torch.equal(model(idx)[0], model(idx)[0])


def test_lr_warms_up_then_glides_down():
    """Starts near 0, peaks after warm-up, ends at min_lr, never goes back up."""
    lrs = [get_lr(s, max_iters=1000, learning_rate=1e-3, min_lr=1e-4,
                  warmup_iters=100) for s in range(1001)]
    assert lrs[0] < 1e-4                      # tiny first step
    assert abs(lrs[99] - 1e-3) < 1e-12        # full rate at the end of warm-up
    assert abs(lrs[1000] - 1e-4) < 1e-12      # min_lr on the last step
    assert all(a <= b for a, b in zip(lrs[:100], lrs[1:100]))     # rising
    assert all(a >= b for a, b in zip(lrs[100:], lrs[101:]))      # falling


def test_on_new_best_only_when_val_improves():
    """The save hook fires only for a val loss lower than every earlier one."""
    torch.manual_seed(config.SEED)
    tok, train_data, val_data = build_dataset(SAMPLE)
    # train() draws batches of config.BLOCK_SIZE, so the window must match.
    model = GPT(tok.vocab_size, n_embd=16, n_head=4, n_layer=2,
                block_size=config.BLOCK_SIZE)
    saved = []
    history = train(model, train_data, val_data, max_iters=200,
                    eval_interval=50, eval_iters=5, min_lr=1e-4,
                    warmup_iters=10, on_new_best=lambda s, v: saved.append((s, v)),
                    device="cpu", verbose=False)
    assert saved[0][0] == 0                                   # first one counts
    assert all(b[1] < a[1] for a, b in zip(saved, saved[1:]))  # each one better
    assert saved[-1][1] == min(v for _, _, v in history)       # best one saved


def test_checkpoint_rebuilds_the_same_model():
    """Save, rebuild from the file alone, and get exactly the same outputs."""
    tok = CharTokenizer(SAMPLE)
    model_args = dict(vocab_size=tok.vocab_size, n_embd=16, n_head=4,
                      n_layer=2, block_size=8, dropout=0.0)
    model = _tiny_gpt(tok.vocab_size)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt", "gpt.pt")
        save_checkpoint(path, model, model_args, tok, step=123, val_loss=1.5)
        ckpt = torch.load(path)
    rebuilt = GPT(**ckpt["model_args"])
    rebuilt.load_state_dict(ckpt["state_dict"])
    model.eval()
    rebuilt.eval()
    idx = torch.tensor([tok.encode("Hello, W")])
    assert torch.equal(model(idx)[0], rebuilt(idx)[0])
    assert CharTokenizer("".join(ckpt["chars"])).chars == tok.chars
    assert ckpt["step"] == 123 and ckpt["val_loss"] == 1.5


def test_plot_writes_a_png():
    """plot_losses creates a real PNG file (checked by its first bytes)."""
    history = [(0, 4.6, 4.6), (100, 2.0, 2.1), (200, 1.5, 1.6)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "plots", "loss.png")
        plot_losses(history, path, baseline=2.38)
        with open(path, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"


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
