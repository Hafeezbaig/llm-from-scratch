"""Stage 5 tests: the temperature and top-k knobs, and generate.py.

Runs two ways -- no test framework required:

    python tests/test_stage5.py     # plain runner
    pytest tests/                   # if you have pytest installed

Everything uses a tiny model on the CPU and a temporary folder, so it is fast,
needs no trained model, and leaves no files behind. To write with the real
model, run ``python generate.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import config  # noqa: E402
from generate import (generate_text, load_model, peek, prepare_prompt,  # noqa: E402
                      real_word_share, variety)
from model import GPT, next_char_probs  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from train import save_checkpoint  # noqa: E402

SAMPLE = "Hello, Watson! The game is afoot.\n" * 100
# Five scores, all different, for one position: the favourite is index 3.
SCORES = torch.tensor([[1.0, -0.5, 2.0, 3.0, 0.5]])


def _tiny_gpt(vocab_size: int) -> GPT:
    torch.manual_seed(config.SEED)
    return GPT(vocab_size, n_embd=16, n_head=4, n_layer=2, block_size=8,
               dropout=0.0).eval()


def test_default_knobs_change_nothing():
    """temperature 1 and no top_k give exactly the model's own odds."""
    assert torch.allclose(next_char_probs(SCORES), F.softmax(SCORES, dim=-1))


def test_temperature_sharpens_or_flattens():
    """Low temperature: the favourite gets likelier. High: less likely.

    The order of the characters never changes, only how far apart they are.
    """
    cold, normal, hot = (next_char_probs(SCORES, t)[0] for t in (0.5, 1.0, 2.0))
    assert cold[3] > normal[3] > hot[3]                   # favourite
    assert cold[1] < normal[1] < hot[1]                   # the long shot
    for p in (cold, normal, hot):
        assert torch.equal(p.argsort(), normal.argsort())  # same order
        assert abs(p.sum().item() - 1) < 1e-6


def test_top_k_keeps_only_the_k_best():
    """top_k=2 leaves exactly the 2 highest scores, and the odds still sum to 1."""
    p = next_char_probs(SCORES, top_k=2)[0]
    assert set(p.nonzero().flatten().tolist()) == {3, 2}
    assert abs(p.sum().item() - 1) < 1e-6
    # A shortlist longer than the alphabet changes nothing.
    assert torch.allclose(next_char_probs(SCORES, top_k=50), F.softmax(SCORES, dim=-1))


def test_bad_knobs_are_refused():
    """temperature must be above 0 and top_k at least 1."""
    for kwargs in ({"temperature": 0}, {"temperature": -1}, {"top_k": 0}):
        try:
            next_char_probs(SCORES, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} should raise ValueError")


def test_top_k_1_always_writes_the_same_text():
    """With only the favourite allowed, the dice no longer matter."""
    model = _tiny_gpt(20)
    start = torch.zeros((1, 1), dtype=torch.long)
    texts = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        texts.append(model.generate(start, 30, top_k=1).tolist())
    assert texts[0] == texts[1] == texts[2]


def test_generate_only_picks_from_the_shortlist():
    """200 dice rolls from the same start with top_k=3 all land in the top 3."""
    model = _tiny_gpt(20)
    start = torch.zeros((200, 1), dtype=torch.long)
    top3 = set(torch.topk(model(start[:1])[0][0, -1], 3).indices.tolist())
    picks = set(model.generate(start, 1, top_k=3)[:, -1].tolist())
    assert picks <= top3 and len(picks) > 1  # stays in the top 3, but still rolls


def test_load_model_then_generate_and_peek():
    """generate.py rebuilds a saved model and continues a prompt with it."""
    tok = CharTokenizer(SAMPLE)
    model_args = dict(vocab_size=tok.vocab_size, n_embd=16, n_head=4,
                      n_layer=2, block_size=8, dropout=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "gpt.pt")
        save_checkpoint(path, _tiny_gpt(tok.vocab_size), model_args, tok,
                        step=1, val_loss=2.0)
        model, loaded_tok = load_model(path, device="cpu")
    text = generate_text(model, loaded_tok, "Hello", 20, temperature=0.8,
                         top_k=5, device="cpu")
    assert text.startswith("Hello") and len(text) == 25
    odds = peek(model, loaded_tok, "Hello", top_k=3, device="cpu")
    assert len(odds) == 3                                    # zeros left out
    assert [p for _, p in odds] == sorted((p for _, p in odds), reverse=True)


def test_missing_checkpoint_is_explained():
    try:
        load_model("no/such/file.pt", device="cpu")
    except FileNotFoundError as exc:
        assert "python train.py" in str(exc)
    else:
        raise AssertionError("a missing checkpoint should raise FileNotFoundError")


def test_prompt_gets_curly_quotes_and_unknown_characters_are_reported():
    tok = CharTokenizer("“”Hi, Watson said he.\n ")
    assert prepare_prompt('"Hi," said he. "Watson"', tok) == "“Hi,” said he. “Watson”"
    try:
        prepare_prompt("Hi @ Watson", tok)
    except ValueError as exc:
        assert "@" in str(exc)
    else:
        raise AssertionError("an unknown character should raise ValueError")


def test_writing_scores():
    """real_word_share counts words the books contain; variety counts repeats."""
    assert real_word_share("The cat, zqx!", {"the", "cat"}) == 2 / 3
    assert variety("the the the cat") == 0.5
    assert real_word_share("", {"the"}) == 0.0 and variety("...") == 0.0


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
