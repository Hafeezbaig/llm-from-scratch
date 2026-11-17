"""Stage 6 tests: the model-size experiment.

Runs two ways -- no test framework required:

    python tests/test_stage6.py     # plain runner
    pytest tests/                   # if you have pytest installed

The experiment is run for only a few steps and writes to a temporary folder, so
this takes seconds and leaves no files behind. The real run is
``python experiments/model_sizes.py`` (best on a cloud GPU).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

# Make the project root importable when this file is run directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from model_sizes import SIZE_COLORS, SIZES, main, plot_sizes  # noqa: E402
from model import GPT  # noqa: E402

QUICK = ["--max-iters", "4", "--eval-interval", "2", "--eval-iters", "1"]


def _run(args: list[str]) -> str:
    """Run the experiment's command line and return what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main(args)
    return out.getvalue()


def test_sizes_grow_and_split_evenly():
    """Each size has more parameters than the one before, and a colour."""
    counts = [sum(p.numel() for p in GPT(102, n_embd=e, n_head=h, n_layer=n,
                                         block_size=8).parameters())
              for _, e, h, n in SIZES]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    assert all(e % h == 0 for _, e, h, _ in SIZES)
    assert len(SIZE_COLORS) == len(SIZES)


def test_plot_writes_a_png():
    fake = [dict(name=name, params=10 ** (5 + i), final_val=2.0 - i / 5,
                 history=[(0, 4.6, 4.6), (100, 2.5, 2.4 - i / 10),
                          (200, 2.2 - i / 5, 2.0 - i / 5)])
            for i, (name, *_) in enumerate(SIZES)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "plots", "sizes.png")
        plot_sizes(fake, path)
        with open(path, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_rerun_skips_finished_sizes():
    """A second run keeps 'tiny' as it was and only trains 'small'."""
    with tempfile.TemporaryDirectory() as tmp:
        files = ["--results", os.path.join(tmp, "r.json"),
                 "--plot", os.path.join(tmp, "p.png")]
        _run(["--sizes", "tiny"] + QUICK + files)
        with open(files[1], encoding="utf-8") as f:
            first = json.load(f)["sizes"]
        printed = _run(["--sizes", "tiny", "small"] + QUICK + files)
        with open(files[1], encoding="utf-8") as f:
            second = json.load(f)["sizes"]
    assert "already done, skipping: tiny" in printed
    assert "=== tiny" not in printed and "=== small" in printed
    assert [r["name"] for r in second] == ["tiny", "small"]
    assert second[0] == first[0]  # tiny's results untouched


def test_changed_recipe_starts_over():
    """Results from a different recipe (here: more steps) are not reused."""
    with tempfile.TemporaryDirectory() as tmp:
        files = ["--results", os.path.join(tmp, "r.json"),
                 "--plot", os.path.join(tmp, "p.png")]
        _run(["--sizes", "tiny"] + QUICK + files)
        more_steps = ["--max-iters", "6"] + QUICK[2:]
        printed = _run(["--sizes", "tiny"] + more_steps + files)
    assert "already done" not in printed and "=== tiny" in printed


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
