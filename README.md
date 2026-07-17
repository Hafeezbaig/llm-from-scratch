# llm-from-scratch

Building a small **GPT-style language model from scratch in PyTorch**, one piece
at a time, to actually understand how modern LLMs work under the hood — not to
beat them.

It reads a large pile of text, learns the character-level patterns in it, and
generates new text in the same style. A tiny **decoder-only transformer** built
without any pre-made model libraries (tokenizer → embeddings → attention →
training → generation).

The model is intentionally small and not "smart" — the point is learning the
machinery, not shipping a product.

**Stack:** Python · PyTorch · matplotlib (loss curves) · trained locally on
Apple Silicon (MPS).

---

## Progress

- [x] **Stage 1 — Data + tokenizer** ← *you are here*
- [ ] Stage 2 — Bigram baseline (a deliberately dumb model to beat)
- [ ] Stage 3 — The transformer (self-attention, multi-head, blocks)
- [ ] Stage 4 — Train properly + loss curves
- [ ] Stage 5 — Generation (temperature, top-k sampling)
- [ ] Stage 6 — Refactor, one experiment, writeup
- [ ] Stage 7 — Live demo (FastAPI backend)

### Stage 1 — done

A complete character-level data pipeline:

- **Dataset** — ~2.7M characters of public-domain **Sherlock Holmes** text from
  Project Gutenberg. See [data/](data/).
- **Tokenizer** — character-level `encode` / `decode` over a 102-character
  vocabulary. See [tokenizer.py](tokenizer.py).
- **Batching** — train/val split plus random `(input, next-character target)`
  batches, placed on the Mac GPU. See [data.py](data.py).
- **Tests** — round-trip and batch shape/alignment checks. See
  [tests/test_stage1.py](tests/test_stage1.py).

---

## Quickstart

```bash
# 1. Set up the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) rebuild the dataset — a copy is already committed
python scripts/prepare_data.py

# 3. See Stage 1 working: tests + a live demo on the real data
python tests/test_stage1.py
```

You should see the tests pass, followed by a demo printing the vocabulary size,
an encode/decode round-trip, and a sample batch mapping each character to its
next-character target.

---

## Project structure (so far)

```
llm-from-scratch/
├── config.py            # all settings in one place (context length, batch size, device)
├── tokenizer.py         # character-level encode / decode
├── data.py              # train/val split + random batching
├── scripts/
│   └── prepare_data.py  # downloads & cleans the dataset (reproducible)
├── tests/
│   └── test_stage1.py   # tokenizer + batching tests, with a live demo
├── data/
│   ├── input.txt        # the training corpus
│   └── README.md        # dataset source & license
├── requirements.txt
└── LICENSE              # MIT
```

More files (`model.py`, `train.py`, `generate.py`, …) arrive in later stages.

---

## License

[MIT](LICENSE) — free to use, learn from, and build on.
Full license details: [hafeezbaig.in/docs/MITLicense](https://www.hafeezbaig.in/docs/MITLicense)

---

Hafeez Baig — [Portfolio](https://hafeezbaig.in)
