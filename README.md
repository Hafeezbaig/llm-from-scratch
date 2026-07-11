# llm-from-scratch

Building a small GPT-style language model from scratch in PyTorch, just to understand how it actually works under the hood.

It reads a bunch of text, learns the patterns, and generates more in the same style.

A tiny decoder-only transformer, coded one piece at a time (tokenizer → embeddings → attention → training → generation). No pre-made model libraries.

The model is meant to be small and not that smart, the point is learning the machinery, not beating real GPT.

**Stack:** Python + PyTorch, matplotlib for loss curves, trained locally on a Mac (MPS).

---

Hafeez Baig - [Portfolio](https://hafeezbaig.in)