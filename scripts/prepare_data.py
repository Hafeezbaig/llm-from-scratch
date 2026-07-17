"""Download and prepare the training corpus.

Fetches several public-domain Sherlock Holmes books from Project Gutenberg,
strips Gutenberg's license header/footer, concatenates them, and writes the
result to ``data/input.txt``.

Run once:

    python scripts/prepare_data.py

The produced ``data/input.txt`` is committed to the repo, so you normally do NOT
need to re-run this. It lives here for reproducibility and to document exactly
where the data came from (everything below is public domain in the US).
"""

from __future__ import annotations

import os
import re
import urllib.request

# (Project Gutenberg ebook id, human-readable title) -- all public domain (US).
BOOKS = [
    (1661, "The Adventures of Sherlock Holmes"),
    (834,  "The Memoirs of Sherlock Holmes"),
    (2097, "The Sign of the Four"),
    (244,  "A Study in Scarlet"),
    (2852, "The Hound of the Baskervilles"),
    (108,  "The Return of Sherlock Holmes"),
]

URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
OUT_PATH = os.path.join("data", "input.txt")

# Gutenberg wraps every book in a standard header/footer we want to discard.
START_RE = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                      re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                    re.IGNORECASE | re.DOTALL)


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Keep only the actual book text, dropping Gutenberg's header and footer."""
    start = START_RE.search(raw)
    if start:
        raw = raw[start.end():]
    end = END_RE.search(raw)
    if end:
        raw = raw[:end.start()]
    # Normalise Windows/Mac line endings to plain "\n" so the committed file is
    # clean LF (Gutenberg ships CRLF, which would otherwise leave stray "\r"s).
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return raw.strip()


def download(book_id: int) -> str:
    """Fetch one book's raw text from Project Gutenberg."""
    url = URL_TEMPLATE.format(id=book_id)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8", errors="replace")


def main() -> None:
    os.makedirs("data", exist_ok=True)
    parts = []
    for book_id, title in BOOKS:
        print(f"Downloading {title} (#{book_id}) ...")
        text = strip_gutenberg_boilerplate(download(book_id))
        print(f"  kept {len(text):,} characters")
        parts.append(text)

    corpus = "\n\n".join(parts) + "\n"
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"\nWrote {len(corpus):,} characters to {OUT_PATH}")


if __name__ == "__main__":
    main()
