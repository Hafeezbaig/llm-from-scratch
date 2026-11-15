"""Stage 2 -- the bigram baseline: a deliberately dumb language model.

A bigram model predicts the next character from ONLY the current character. It
has no memory of anything that came before. Under the hood it is a single
lookup table: one row per character, and each row holds a score for every
possible next character.

It exists for two reasons:

1. To get the whole loop working end to end:
   data -> model -> loss -> train -> generate.
2. To give us a baseline loss that the transformer (Stage 3) has to beat.

Run it:

    python bigram.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import config
from data import build_dataset, get_batch


class BigramLanguageModel(nn.Module):
    """Next-character scores read straight out of one lookup table.

    The table is ``vocab_size x vocab_size``. Row ``i`` holds the scores
    ("logits") for which character comes after character ``i``. Training nudges
    those scores so the character that really comes next gets a higher score.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Score every possible next character, plus the loss if given targets.

        ``idx`` and ``targets`` are ``(B, T)`` tensors of ids (batch, time).
        Returns ``(logits, loss)``: logits are ``(B, T, vocab_size)``, and loss
        is ``None`` when there are no targets (e.g. while generating).
        """
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        # cross_entropy wants one row per prediction, so flatten batch and time.
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Extend ``idx`` (B, T) by ``max_new_tokens`` sampled characters.

        Each step: score the next character, turn the scores into probabilities,
        draw one character at random using those probabilities, and append it.
        A bigram only needs the last character, but we pass the whole sequence
        so the same loop works for the transformer later.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                           # (B, C) last position only
            probs = F.softmax(logits, dim=-1)                   # scores -> probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) weighted dice roll
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T + 1)
        return idx


@torch.no_grad()
def estimate_loss(model: nn.Module,
                  train_data: torch.Tensor,
                  val_data: torch.Tensor,
                  eval_iters: int = config.EVAL_ITERS,
                  device: str = config.DEVICE) -> dict[str, float]:
    """Average the loss over ``eval_iters`` random batches of each split.

    One batch's loss is noisy, so averaging many gives a steadier number.
    """
    model.eval()  # no effect on the bigram, but the transformer will need it
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


def main() -> None:
    torch.manual_seed(config.SEED)
    tokenizer, train_data, val_data = build_dataset()
    model = BigramLanguageModel(tokenizer.vocab_size).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {config.DEVICE} | vocab {tokenizer.vocab_size}"
          f" | parameters {n_params:,}")
    # The score to beat on day one: guessing evenly among every character.
    print(f"loss of an even guess among {tokenizer.vocab_size} characters:"
          f" {math.log(tokenizer.vocab_size):.4f}")

    # Every sample starts from id 0, the newline character.
    start = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)

    print("\n--- sample BEFORE training (random table) ---")
    print(tokenizer.decode(model.generate(start, max_new_tokens=300)[0].tolist()))

    print("\n--- training ---")
    train(model, train_data, val_data)

    print("\n--- sample AFTER training ---")
    print(tokenizer.decode(model.generate(start, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
