"""Stage 3 -- the transformer: a small GPT, built piece by piece.

The pieces, in the order they are built (each one uses the ones above it):

1. ``Head``               -- one head of self-attention
2. ``MultiHeadAttention`` -- several heads side by side
3. ``FeedForward``        -- a small two-layer network, run at every position
4. ``Block``              -- attention + feed-forward, with residuals and layer norm
5. ``GPT``                -- token + position embeddings -> blocks -> next-char scores

The bigram only ever sees the current character. Here, every position can look
back at all the earlier characters in its window (up to ``BLOCK_SIZE``) and
decide which of them matter for guessing the next one. That looking back is
self-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

import config


class Head(nn.Module):
    """One head of causal self-attention.

    Every position turns its vector into three smaller ones:

    * query -- "what am I looking for?"
    * key   -- "what do I have?"
    * value -- "what I pass on if you pick me"

    Matching a position's query against every key gives a score for how useful
    each other position is. Future positions are masked out (a position must not
    see the character it is trying to predict), the scores become weights with
    softmax, and the output is the weighted mix of the values.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular ones: row t has ones in columns 0..t, so position t
        # may only look at itself and earlier positions. A buffer moves to the
        # device with the model but is never trained.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # Row t, column s = how much position t wants to look at position s.
        # Dividing by sqrt(head_size) keeps the scores small enough that softmax
        # spreads its attention instead of locking onto a single position.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # each row now sums to 1
        v = self.value(x)  # (B, T, head_size)
        return wei @ v     # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, their outputs joined back together.

    Each head can learn to look for a different kind of pattern. Their outputs
    are concatenated and projected back to ``n_embd`` so the block can add the
    result to its input.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            Head(n_embd, head_size, block_size) for _ in range(n_head))
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.proj(out)


class FeedForward(nn.Module):
    """A small two-layer network, applied to each position on its own.

    Attention moves information between positions; this is where each position
    then works on what it gathered. The hidden layer is 4x wider, as in the
    original transformer paper.
    """

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate (attention), then compute (feed-forward).

    Two details make a deep stack of blocks trainable:

    * Residual connections (``x = x + ...``): each part only adds a correction
      to its input instead of replacing it, so information and training signal
      can flow straight through the stack.
    * Layer norm *before* each part ("pre-norm", as in GPT-2): keeps the numbers
      at every position at a steady scale.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full decoder-only transformer.

    ids -> token embedding + position embedding -> ``n_layer`` blocks -> final
    layer norm -> a linear layer that scores every possible next character.

    It has the same interface as the bigram (``forward(idx, targets)`` and
    ``generate``), so the same training loop drives both.
    """

    def __init__(self,
                 vocab_size: int,
                 n_embd: int = config.N_EMBD,
                 n_head: int = config.N_HEAD,
                 n_layer: int = config.N_LAYER,
                 block_size: int = config.BLOCK_SIZE) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must split evenly across the heads"
        self.block_size = block_size
        # One learned vector for WHICH character it is, one for WHERE it sits.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Score every possible next character, plus the loss if given targets.

        ``idx`` and ``targets`` are ``(B, T)`` tensors of ids, with ``T`` at most
        ``block_size``. Returns ``(logits, loss)``: logits are
        ``(B, T, vocab_size)``, and loss is ``None`` when there are no targets.
        """
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)    # 0, 1, ..., T-1
        tok_emb = self.token_embedding_table(idx)         # (B, T, n_embd)
        pos_emb = self.position_embedding_table(positions)  # (T, n_embd)
        x = tok_emb + pos_emb   # (B, T, n_embd): same positions for every row
        x = self.blocks(x)      # (B, T, n_embd)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Extend ``idx`` (B, T) by ``max_new_tokens`` sampled characters."""
        for _ in range(max_new_tokens):
            # The position table only has block_size rows, so the model can only
            # look at the last block_size characters.
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                           # (B, C) last position only
            probs = F.softmax(logits, dim=-1)                   # scores -> probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1) weighted dice roll
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T + 1)
        return idx
