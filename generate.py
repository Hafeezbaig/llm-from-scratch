"""Stage 5 -- write text with the trained model, and control how bold it is.

Loads the model that ``train.py`` saved (``checkpoints/gpt.pt``), so there is no
training here: writing a few hundred characters takes seconds.

Two knobs change how each next character is picked (see ``next_char_probs`` in
``model.py``):

* ``--temperature`` -- below 1 is careful (tidy but repetitive), above 1 is bold
  (inventive but messy). 1 keeps the model's own odds.
* ``--top_k`` -- only the k most likely characters may be picked, so a wild
  long shot can never slip in. ``--top_k 1`` always picks the favourite.

Run it:

    python generate.py                                    # 500 characters
    python generate.py --prompt "Holmes" --temperature 0.5 --top_k 10
    python generate.py --peek "the game is a"             # show the odds for one pick
    python generate.py --compare                          # same start, several settings
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

import config
from model import GPT, next_char_probs
from tokenizer import CharTokenizer

# The settings ``--compare`` writes with, from most careful to most bold.
COMPARE_SETTINGS = [
    ("always the favourite", 1.0, 1),
    ("careful", 0.5, None),
    ("a bit careful (default)", 0.8, None),
    ("the model's own odds", 1.0, None),
    ("bold", 1.5, None),
    ("wild", 2.0, None),
    ("wild + shortlist of 5", 2.0, 5),
]


def load_model(path: str = config.CHECKPOINT_PATH,
               device: str = config.DEVICE) -> tuple[GPT, CharTokenizer]:
    """Rebuild the trained model and its tokenizer from a checkpoint file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at {path}. Train one first: python train.py")
    ckpt = torch.load(path, map_location=device)
    model = GPT(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()  # eval: dropout off while writing
    return model, CharTokenizer("".join(ckpt["chars"]))


def prepare_prompt(text: str, tokenizer: CharTokenizer) -> str:
    """Make a typed prompt usable by the model, or explain why it can't be.

    The books use curly quotes (“ ”), which a keyboard doesn't type, so straight
    double quotes are swapped for curly ones: an opening quote at the start or
    after a space, a closing quote anywhere else. Any other character the model
    never saw can't be encoded, so it's reported instead of crashing.
    """
    text = re.sub(r'(^|(?<=\s))"', "“", text)
    text = text.replace('"', "”")
    unknown = sorted(set(text) - set(tokenizer.chars))
    if unknown:
        raise ValueError(f"the model never saw these characters: {unknown}. "
                         "Leave them out or swap them for similar ones.")
    return text


def generate_text(model: GPT,
                  tokenizer: CharTokenizer,
                  prompt: str = "",
                  max_new_tokens: int = 500,
                  temperature: float = 1.0,
                  top_k: int | None = None,
                  device: str = config.DEVICE) -> str:
    """Continue ``prompt`` by ``max_new_tokens`` characters; return all of it.

    With no prompt, it starts from a newline, as the training samples do.
    """
    start = prepare_prompt(prompt, tokenizer) or "\n"
    idx = torch.tensor([tokenizer.encode(start)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


@torch.no_grad()
def peek(model: GPT,
         tokenizer: CharTokenizer,
         prompt: str,
         temperature: float = 1.0,
         top_k: int | None = None,
         n: int = 8,
         device: str = config.DEVICE) -> list[tuple[str, float]]:
    """The ``n`` likeliest next characters after ``prompt``, with their odds.

    These are the odds after the two knobs are applied: exactly what the dice
    roll would use for the next character. Characters ``top_k`` ruled out
    (odds 0) are left out.
    """
    text = prepare_prompt(prompt, tokenizer) or "\n"
    idx = torch.tensor([tokenizer.encode(text)[-model.block_size:]],
                       dtype=torch.long, device=device)
    logits, _ = model(idx)
    probs = next_char_probs(logits[:, -1, :], temperature, top_k)[0]
    top = torch.topk(probs, min(n, probs.numel()))
    return [(tokenizer.decode([i]), p)
            for p, i in zip(top.values.tolist(), top.indices.tolist())
            if p > 0]  # top_k can leave fewer than n possible characters


# --- Scoring a piece of writing -------------------------------------------------
def words(text: str) -> list[str]:
    """Lower-case runs of letters: "Don't, Watson!" -> ["don", "t", "watson"]."""
    return re.findall(r"[a-z]+", text.lower())


def real_word_share(text: str, known_words: set[str]) -> float:
    """Fraction of the words in ``text`` that also appear in the books."""
    ws = words(text)
    return sum(w in known_words for w in ws) / len(ws) if ws else 0.0


def variety(text: str) -> float:
    """Distinct words / all words. Low means the text keeps repeating itself.

    It drops for longer texts too, so only compare texts of the same length.
    """
    ws = words(text)
    return len(set(ws)) / len(ws) if ws else 0.0


# --- The command line -----------------------------------------------------------
def show_char(ch: str) -> str:
    return {" ": "space", "\n": "newline"}.get(ch, ch)


def knobs(temperature: float, top_k: int | None) -> str:
    return f"temperature {temperature}, top_k {top_k if top_k else 'off'}"


def print_peek(model: GPT, tokenizer: CharTokenizer, prompt: str,
               temperature: float, top_k: int | None) -> None:
    shown = prepare_prompt(prompt, tokenizer)
    print(f"Next character after {shown!r} ({knobs(temperature, top_k)}):\n")
    for ch, p in peek(model, tokenizer, prompt, temperature, top_k):
        bar = "█" * round(p * 40)
        print(f"  {show_char(ch):>7}  {p * 100:5.1f}%  {bar}")


def print_compare(model: GPT, tokenizer: CharTokenizer, prompt: str,
                  length: int, seed: int) -> None:
    """Write from the same start with every setting in ``COMPARE_SETTINGS``.

    Each setting writes 4 pieces of ``length`` characters (one batch) from the
    same seed. The first piece is printed, and all 4 are scored.
    """
    with open(config.DATA_PATH, encoding="utf-8") as f:
        known = set(words(f.read()))
    start = prepare_prompt(prompt, tokenizer) or "\n"
    rows = []
    for name, temperature, top_k in COMPARE_SETTINGS:
        torch.manual_seed(seed)
        idx = torch.tensor([tokenizer.encode(start)] * 4, dtype=torch.long,
                           device=config.DEVICE)
        out = model.generate(idx, length, temperature=temperature, top_k=top_k)
        texts = [tokenizer.decode(row[len(start):].tolist()) for row in out]
        real = sum(real_word_share(t, known) for t in texts) / len(texts)
        var = sum(variety(t) for t in texts) / len(texts)
        rows.append((name, temperature, top_k, real, var))
        print(f"=== {name} ({knobs(temperature, top_k)}) ===")
        print((start + texts[0][:400]).strip("\n"))
        print(f"--- real words {real * 100:.0f}% | variety {var * 100:.0f}%\n")

    print(f"Summary (4 pieces of {length} characters each, seed {seed}):")
    print(f"  {'setting':<24} {'temp':>5} {'top_k':>6} {'real words':>11} {'variety':>8}")
    for name, temperature, top_k, real, var in rows:
        print(f"  {name:<24} {temperature:>5} {str(top_k or 'off'):>6}"
              f" {real * 100:>10.0f}% {var * 100:>7.0f}%")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write text with the trained model (checkpoints/gpt.pt).")
    parser.add_argument("--prompt", default="",
                        help="text to start from (default: a new line)")
    parser.add_argument("--tokens", type=int, default=500,
                        help="how many characters to write (default: 500)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="below 1 = careful, above 1 = bold (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="only pick from the k likeliest characters (default: off)")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="same seed + same settings = same text")
    parser.add_argument("--peek", metavar="TEXT",
                        help="show the odds for the next character after TEXT, then stop")
    parser.add_argument("--compare", action="store_true",
                        help="write with several settings side by side, with scores")
    parser.add_argument("--checkpoint", default=config.CHECKPOINT_PATH)
    args = parser.parse_args(argv)

    try:
        model, tokenizer = load_model(args.checkpoint)
        torch.manual_seed(args.seed)
        if args.peek is not None:
            print_peek(model, tokenizer, args.peek, args.temperature, args.top_k)
        elif args.compare:
            print_compare(model, tokenizer, args.prompt, args.tokens, args.seed)
        else:
            print(generate_text(model, tokenizer, args.prompt, args.tokens,
                                args.temperature, args.top_k).strip("\n"))
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
