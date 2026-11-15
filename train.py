"""The training loop, and the script that trains the Stage 3 transformer.

``estimate_loss``, ``train`` and ``sample`` work with any model that has the
same interface (``forward(idx, targets) -> (logits, loss)`` and ``generate``),
so the Stage 2 bigram uses them too.

Run it:

    python train.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

import config
from data import build_dataset, get_batch
from model import GPT
from tokenizer import CharTokenizer


@torch.no_grad()
def estimate_loss(model: nn.Module,
                  train_data: torch.Tensor,
                  val_data: torch.Tensor,
                  eval_iters: int = config.EVAL_ITERS,
                  device: str = config.DEVICE) -> dict[str, float]:
    """Average the loss over ``eval_iters`` random batches of each split.

    One batch's loss is noisy, so averaging many gives a steadier number.
    """
    model.eval()
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


def train(model: nn.Module,
          train_data: torch.Tensor,
          val_data: torch.Tensor,
          max_iters: int = config.MAX_ITERS,
          eval_interval: int = config.EVAL_INTERVAL,
          eval_iters: int = config.EVAL_ITERS,
          learning_rate: float = config.LEARNING_RATE,
          device: str = config.DEVICE,
          verbose: bool = True) -> list[tuple[int, float, float]]:
    """Run the training loop and return ``[(step, train_loss, val_loss), ...]``.

    Every step: grab a batch, measure the loss, work out which way each number
    in the model should move to lower it (``backward``), then move it a little
    (``optimizer.step``). The first history entry is the untrained model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []

    def evaluate(step: int) -> None:
        losses = estimate_loss(model, train_data, val_data, eval_iters, device)
        history.append((step, losses["train"], losses["val"]))
        if verbose:
            print(f"step {step:5d} | train loss {losses['train']:.4f}"
                  f" | val loss {losses['val']:.4f}")

    for step in range(max_iters):
        if step % eval_interval == 0:
            evaluate(step)
        x, y = get_batch(train_data, device=device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    evaluate(max_iters)
    return history


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
    model = GPT(tokenizer.vocab_size).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {config.DEVICE} | vocab {tokenizer.vocab_size}"
          f" | parameters {n_params:,}")
    print(f"{config.N_LAYER} blocks | {config.N_HEAD} heads | embedding"
          f" {config.N_EMBD} | context {config.BLOCK_SIZE} characters")

    print("\n--- sample BEFORE training ---")
    print(sample(model, tokenizer, max_new_tokens=300))

    print("\n--- training ---")
    start = time.time()
    train(model, train_data, val_data)
    print(f"training took {time.time() - start:.0f}s")

    print("\n--- sample AFTER training ---")
    print(sample(model, tokenizer, max_new_tokens=500))


if __name__ == "__main__":
    main()
