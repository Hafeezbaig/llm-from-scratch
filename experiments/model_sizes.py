"""Stage 6 experiment: does a bigger brain help, and by how much?

Trains four sizes of the same GPT with exactly the same recipe and compares
them. Only the three size settings change (``n_embd``, ``n_head``,
``n_layer``). Everything else is shared: the data, context length, batch size,
dropout, learning-rate schedule, number of steps, and even the order of the
training batches (the random seed is reset after each model is built).

Writes:

* ``experiments/results/model_sizes.json`` -- every number: loss curves,
  parameter counts, timings and a writing sample from each size
* ``assets/model_sizes.png`` -- the graph

It never touches ``checkpoints/gpt.pt``, the model ``generate.py`` uses.

The results file is saved after every size, and sizes already in it (trained
with the same recipe) are skipped. So if a run stops halfway, for example when
a cheap "spot" cloud machine is taken back, running it again only trains the
sizes that are missing. Use ``--fresh`` to start over.

Run it from the project root (about 17 minutes of training on a cloud L4 GPU;
roughly 30 minutes on an M5 Mac at full load, which gets hot):

    python experiments/model_sizes.py
    python experiments/model_sizes.py --plot-only   # redraw from the saved JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make the project root importable when this file is run directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw straight to a file; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from data import build_dataset  # noqa: E402
from model import GPT  # noqa: E402
from train import BIGRAM_VAL_LOSS, train  # noqa: E402

RESULTS_PATH = os.path.join("experiments", "results", "model_sizes.json")
PLOT_PATH = os.path.join("assets", "model_sizes.png")

# (name, n_embd, n_head, n_layer), smallest first. Each is several times
# bigger than the one before it.
SIZES = [
    ("tiny", 64, 4, 2),
    ("small", 128, 4, 4),    # Stage 3's size
    ("medium", 256, 4, 4),
    ("big", 384, 6, 6),      # Stage 4's size
]

# One blue per size, light to dark (an ordered colour scale for an ordered
# quantity). The lightest is still dark enough to see on the light background.
SIZE_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def run_size(name: str, n_embd: int, n_head: int, n_layer: int,
             train_data: torch.Tensor, val_data: torch.Tensor, tokenizer,
             max_iters: int, eval_interval: int, eval_iters: int) -> dict:
    """Train one size with the shared recipe and return its results."""
    torch.manual_seed(config.SEED)          # same starting numbers every run
    model = GPT(tokenizer.vocab_size, n_embd=n_embd, n_head=n_head,
                n_layer=n_layer, block_size=config.BLOCK_SIZE,
                dropout=config.DROPOUT).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    torch.manual_seed(config.SEED + 1)      # same batches, in the same order
    print(f"\n=== {name}: {n_params:,} parameters "
          f"({n_layer} blocks, {n_head} heads, embedding {n_embd}) ===", flush=True)

    start = time.time()
    history = train(model, train_data, val_data, max_iters=max_iters,
                    eval_interval=eval_interval, eval_iters=eval_iters,
                    min_lr=config.MIN_LR, warmup_iters=config.WARMUP_ITERS)
    minutes = (time.time() - start) / 60

    # A writing sample at the Stage 5 default boldness, the same for every size.
    model.eval()
    torch.manual_seed(config.SEED)
    begin = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)
    sample = tokenizer.decode(model.generate(begin, 300, temperature=0.8)[0, 1:].tolist())

    _, final_train, final_val = history[-1]
    print(f"{name}: final train {final_train:.4f} | val {final_val:.4f}"
          f" | {minutes:.1f} min", flush=True)
    hardware = (torch.cuda.get_device_name() if config.DEVICE == "cuda"
                else config.DEVICE)
    return dict(name=name, n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                params=n_params, hardware=hardware, minutes=round(minutes, 2),
                final_train=final_train, final_val=final_val,
                history=history, sample=sample)


def plot_sizes(results: list[dict], path: str,
               baseline: float = BIGRAM_VAL_LOSS) -> None:
    """Two panels: val loss while training, and final val loss against size."""
    ink, ink_2, muted, grid = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    surface = "#fcfcfb"
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150,
                                      gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.patch.set_facecolor(surface)

    # Left: how the exam score (val loss) fell while each size studied.
    for r, color in zip(results, SIZE_COLORS):
        steps = [s for s, _, _ in r["history"]]
        vals = [v for _, _, v in r["history"]]
        left.plot(steps, vals, color=color, linewidth=2, label=r["name"])
        left.annotate(f"{r['name']} {vals[-1]:.2f}", (steps[-1], vals[-1]),
                      xytext=(6, 0), textcoords="offset points", va="center",
                      color=ink_2, fontsize=9)
    last_step = results[0]["history"][-1][0]
    left.axhline(baseline, color=muted, linewidth=1.5, linestyle=(0, (4, 3)))
    left.annotate(f"bigram baseline ({baseline:.2f})", (last_step * 0.25, baseline),
                  xytext=(0, 5), textcoords="offset points", ha="left",
                  va="bottom", color=ink_2, fontsize=9)
    left.set_ylim(1.0, 2.6)       # the step-0 scores (~4.7) sit above the top
    left.set_xlim(0, last_step * 1.2)  # room for the end labels
    left.set_title("Validation loss while training", loc="left", color=ink,
                   fontsize=12, pad=18)
    left.text(0, 1.02, "Same recipe for every size. Lower is better.",
              transform=left.transAxes, color=ink_2, fontsize=9)
    left.set_xlabel("training step", color=ink_2, fontsize=9)
    left.set_ylabel("val loss", color=ink_2, fontsize=9)
    # Below the dashed baseline, in the empty space above the curves.
    left.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.82),
                fontsize=9, labelcolor=ink_2)

    # Right: the final score against the number of parameters.
    params = [r["params"] for r in results]
    finals = [r["final_val"] for r in results]
    right.plot(params, finals, color=muted, linewidth=1.5, zorder=1)
    for r, color in zip(results, SIZE_COLORS):
        right.scatter(r["params"], r["final_val"], s=70, color=color,
                      edgecolors=surface, linewidths=2, zorder=2)
        # Above-right of each dot: the line runs down and to the right, so
        # that corner is always clear of it.
        right.annotate(f"{r['name']}\n{r['params'] / 1e6:.1f}M · {r['final_val']:.2f}",
                       (r["params"], r["final_val"]), xytext=(8, 6),
                       textcoords="offset points", ha="left", va="bottom",
                       color=ink_2, fontsize=9)
    right.set_xscale("log")
    right.set_xlim(min(params) / 2.5, max(params) * 6)  # room for the last label
    low, high = min(finals), max(finals)
    right.set_ylim(low - 0.08, high + (high - low) * 0.35)  # room for the labels
    right.set_title("Final validation loss by size", loc="left", color=ink,
                    fontsize=12, pad=18)
    right.text(0, 1.02, "Each step right is several times more parameters.",
               transform=right.transAxes, color=ink_2, fontsize=9)
    right.set_xlabel("parameters (log scale)", color=ink_2, fontsize=9)

    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=muted, labelsize=9)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the graph from the saved results, no training")
    parser.add_argument("--sizes", nargs="+", default=[s[0] for s in SIZES],
                        choices=[s[0] for s in SIZES], help="which sizes to train")
    parser.add_argument("--max-iters", type=int, default=config.MAX_ITERS)
    parser.add_argument("--eval-interval", type=int, default=config.EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=config.EVAL_ITERS)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore saved results and train every size again")
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--plot", default=PLOT_PATH)
    args = parser.parse_args(argv)

    if args.plot_only:
        with open(args.results, encoding="utf-8") as f:
            plot_sizes(json.load(f)["sizes"], args.plot)
        print(f"graph saved to {args.plot}")
        return

    recipe = dict(max_iters=args.max_iters, eval_interval=args.eval_interval,
                  eval_iters=args.eval_iters, block_size=config.BLOCK_SIZE,
                  batch_size=config.BATCH_SIZE, dropout=config.DROPOUT,
                  learning_rate=config.LEARNING_RATE, min_lr=config.MIN_LR,
                  warmup_iters=config.WARMUP_ITERS, seed=config.SEED)
    results = []
    if not args.fresh and os.path.exists(args.results):
        with open(args.results, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("recipe") == recipe:  # only reuse a run with the same recipe
            results = [r for r in saved["sizes"] if r["name"] in args.sizes]
    done = {r["name"] for r in results}
    if done:
        print(f"already done, skipping: {', '.join(sorted(done))}"
              f" (from {args.results}; --fresh to redo)")

    tokenizer, train_data, val_data = build_dataset()
    print(f"device {config.DEVICE} | {args.max_iters} steps per size"
          f" | sizes: {', '.join(args.sizes)}")
    order = [s[0] for s in SIZES]
    for name, n_embd, n_head, n_layer in SIZES:
        if name not in args.sizes or name in done:
            continue
        results.append(run_size(name, n_embd, n_head, n_layer, train_data,
                                val_data, tokenizer, args.max_iters,
                                args.eval_interval, args.eval_iters))
        results.sort(key=lambda r: order.index(r["name"]))
        # Save after every size, so a stopped run still keeps what it finished.
        os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
        with open(args.results, "w", encoding="utf-8") as f:
            json.dump(dict(recipe=recipe, sizes=results), f, indent=1,
                      ensure_ascii=False)

    plot_sizes(results, args.plot)
    print(f"\nresults saved to {args.results} | graph saved to {args.plot}")
    print(f"\n{'size':<8} {'parameters':>12} {'train':>7} {'val':>7} {'minutes':>8}")
    for r in results:
        print(f"{r['name']:<8} {r['params']:>12,} {r['final_train']:>7.3f}"
              f" {r['final_val']:>7.3f} {r['minutes']:>8.1f}")


if __name__ == "__main__":
    main()
