"""The training loop, and the script that trains the transformer.

``estimate_loss``, ``train`` and ``sample`` work with any model that has the
same interface (``forward(idx, targets) -> (logits, loss)`` and ``generate``),
so the Stage 2 bigram uses them too.

Stage 4 added what "training properly" needs: a learning rate that warms up and
then glides down (``get_lr``), saving the best model so far to disk
(``save_checkpoint``), and a picture of the loss curves (``plot_losses``).

Run it:

    python train.py
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # draw straight to a file; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import config  # noqa: E402
from data import build_dataset, get_batch  # noqa: E402
from model import GPT  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

# Stage 2's final val loss (``python bigram.py``): the score to beat, drawn on
# the loss plot for comparison.
BIGRAM_VAL_LOSS = 2.38


@torch.no_grad()
def estimate_loss(model: nn.Module,
                  train_data: torch.Tensor,
                  val_data: torch.Tensor,
                  eval_iters: int = config.EVAL_ITERS,
                  device: str = config.DEVICE) -> dict[str, float]:
    """Average the loss over ``eval_iters`` random batches of each split.

    One batch's loss is noisy, so averaging many gives a steadier number.
    """
    model.eval()  # dropout off while measuring
    out = {}
    for name, split_data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_data, device=device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def get_lr(step: int, max_iters: int, learning_rate: float, min_lr: float,
           warmup_iters: int) -> float:
    """The learning rate for one step: warm up, then glide down.

    * For the first ``warmup_iters`` steps, rise in a straight line from almost 0
      to ``learning_rate``, so big early steps don't knock the random, untrained
      model around.
    * Then follow half a cosine wave from ``learning_rate`` down to ``min_lr``:
      big steps while there is a lot to learn, small careful steps at the end.
    """
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)  # 0 -> 1
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_lr)


def train(model: nn.Module,
          train_data: torch.Tensor,
          val_data: torch.Tensor,
          max_iters: int = config.MAX_ITERS,
          eval_interval: int = config.EVAL_INTERVAL,
          eval_iters: int = config.EVAL_ITERS,
          learning_rate: float = config.LEARNING_RATE,
          min_lr: float | None = None,
          warmup_iters: int = 0,
          on_new_best: Callable[[int, float], None] | None = None,
          device: str = config.DEVICE,
          verbose: bool = True) -> list[tuple[int, float, float]]:
    """Run the training loop and return ``[(step, train_loss, val_loss), ...]``.

    Every step: grab a batch, measure the loss, work out which way each number
    in the model should move to lower it (``backward``), then move it a little
    (``optimizer.step``). The first history entry is the untrained model.

    With ``min_lr`` set, the learning rate follows ``get_lr``; without it, it
    stays at ``learning_rate`` the whole time (as the bigram does).
    ``on_new_best(step, val_loss)`` is called whenever the val loss beats every
    earlier measurement -- the place to save a checkpoint.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    best_val = float("inf")

    def evaluate(step: int) -> None:
        nonlocal best_val
        losses = estimate_loss(model, train_data, val_data, eval_iters, device)
        history.append((step, losses["train"], losses["val"]))
        if verbose:
            print(f"step {step:5d} | train loss {losses['train']:.4f}"
                  f" | val loss {losses['val']:.4f}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            if on_new_best is not None:
                on_new_best(step, best_val)

    for step in range(max_iters):
        if step % eval_interval == 0:
            evaluate(step)
        if min_lr is not None:
            lr = get_lr(step, max_iters, learning_rate, min_lr, warmup_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr
        x, y = get_batch(train_data, device=device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    evaluate(max_iters)
    return history


def save_checkpoint(path: str,
                    model: nn.Module,
                    model_args: dict,
                    tokenizer: CharTokenizer,
                    step: int,
                    val_loss: float) -> None:
    """Save everything needed to rebuild the trained model without retraining.

    The weights alone are not enough: we also store the settings the model was
    built with (to rebuild the same shape) and the tokenizer's characters (so
    ids map back to the same letters).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_args": model_args,
        "state_dict": model.state_dict(),
        "chars": tokenizer.chars,
        "step": step,
        "val_loss": val_loss,
    }, path)


def plot_losses(history: list[tuple[int, float, float]],
                path: str,
                baseline: float | None = None) -> None:
    """Draw train and val loss against training step, and save it as a PNG."""
    steps = [s for s, _, _ in history]
    train_losses = [t for _, t, _ in history]
    val_losses = [v for _, _, v in history]

    ink, ink_2, muted = "#0b0b0b", "#52514e", "#898781"
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(steps, train_losses, color="#2a78d6", linewidth=2, label="train loss")
    ax.plot(steps, val_losses, color="#eb6834", linewidth=2, label="val loss")
    if baseline is not None:
        ax.axhline(baseline, color=muted, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.annotate(f"bigram baseline ({baseline:.2f})", (steps[-1], baseline),
                    xytext=(0, 5), textcoords="offset points", ha="right",
                    va="bottom", color=ink_2, fontsize=9)

    # Direct labels at the line ends, nudged apart so they never overlap.
    train_end, val_end = train_losses[-1], val_losses[-1]
    for value, name, higher in ((train_end, "train", train_end >= val_end),
                                (val_end, "val", val_end > train_end)):
        ax.annotate(f"{name} {value:.2f}", (steps[-1], value),
                    xytext=(6, 5 if higher else -5), textcoords="offset points",
                    va="bottom" if higher else "top", color=ink_2, fontsize=9)

    ax.set_title("Training and validation loss", loc="left", color=ink,
                 fontsize=12, pad=18)
    ax.text(0, 1.02, "Lower is better. Val = text the model never trains on.",
            transform=ax.transAxes, color=ink_2, fontsize=9)
    ax.set_xlabel("training step", color=ink_2, fontsize=9)
    ax.set_ylabel("loss", color=ink_2, fontsize=9)
    ax.set_xlim(0, steps[-1] * 1.1)  # room for the end labels
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=muted, labelsize=9)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=ink_2)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def sample(model: nn.Module,
           tokenizer: CharTokenizer,
           max_new_tokens: int,
           device: str = config.DEVICE) -> str:
    """Generate text starting from id 0, the newline character."""
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    text = tokenizer.decode(model.generate(start, max_new_tokens)[0].tolist())
    model.train()
    return text


def main() -> None:
    torch.manual_seed(config.SEED)
    tokenizer, train_data, val_data = build_dataset()
    model_args = dict(vocab_size=tokenizer.vocab_size, n_embd=config.N_EMBD,
                      n_head=config.N_HEAD, n_layer=config.N_LAYER,
                      block_size=config.BLOCK_SIZE, dropout=config.DROPOUT)
    model = GPT(**model_args).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {config.DEVICE} | vocab {tokenizer.vocab_size}"
          f" | parameters {n_params:,}")
    print(f"{config.N_LAYER} blocks | {config.N_HEAD} heads | embedding"
          f" {config.N_EMBD} | context {config.BLOCK_SIZE} characters"
          f" | dropout {config.DROPOUT}")

    print("\n--- sample BEFORE training ---")
    print(sample(model, tokenizer, max_new_tokens=300))

    def save_best(step: int, val_loss: float) -> None:
        save_checkpoint(config.CHECKPOINT_PATH, model, model_args, tokenizer,
                        step, val_loss)
        print(f"        new best val loss -> saved {config.CHECKPOINT_PATH}")

    print("\n--- training ---")
    start = time.time()
    history = train(model, train_data, val_data, min_lr=config.MIN_LR,
                    warmup_iters=config.WARMUP_ITERS, on_new_best=save_best)
    print(f"training took {(time.time() - start) / 60:.1f} min")

    plot_losses(history, config.LOSS_PLOT_PATH, baseline=BIGRAM_VAL_LOSS)
    print(f"loss curve saved to {config.LOSS_PLOT_PATH}")

    print("\n--- sample AFTER training ---")
    print(sample(model, tokenizer, max_new_tokens=1000))


if __name__ == "__main__":
    main()
