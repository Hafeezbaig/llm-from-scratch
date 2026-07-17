# Dataset

The training corpus is [`input.txt`](input.txt): the text of six public-domain
**Sherlock Holmes** books by Arthur Conan Doyle, downloaded from
[Project Gutenberg](https://www.gutenberg.org/) and concatenated into one file.

- ~2.7 million characters of Victorian English prose (dialogue-heavy, which
  makes the model's later output recognisably "Holmes-ish").
- Every title is in the **public domain** in the United States.
- Project Gutenberg's license header/footer is stripped out; only the book text
  remains.

## Regenerate

`input.txt` is committed to the repo, so you normally don't need to. To rebuild
it from scratch:

```bash
python scripts/prepare_data.py
```

## Books included

| Gutenberg # | Title |
|---|---|
| 1661 | The Adventures of Sherlock Holmes |
| 834  | The Memoirs of Sherlock Holmes |
| 2097 | The Sign of the Four |
| 244  | A Study in Scarlet |
| 2852 | The Hound of the Baskervilles |
| 108  | The Return of Sherlock Holmes |
