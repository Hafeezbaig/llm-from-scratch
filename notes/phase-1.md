# Stage 1, explained simply

## The big idea

A language model plays one guessing game: **look at some text and guess the next character.** If it gets very good at this, it can write by guessing one character, adding it to the text, and guessing again.

Stage 1 did not build that model, so **nothing has learned anything yet**. It only prepared the study material, like making flashcards before the student arrives.

```
6 books → one text file → numbers → 90% study / 10% test → random flashcards
```

## Step 1: Get the text
[prepare_data.py](../scripts/prepare_data.py) downloads six Sherlock Holmes books and removes the website's legal text from each one. It then joins them into one file, [input.txt](../data/input.txt), with 2,631,112 characters.

The model will learn to write in this style, because this is all it ever reads.

## Step 2: Turn characters into numbers
Computers can only do maths, so each character needs a number. [tokenizer.py](../tokenizer.py) does this in three steps:

1. It finds every different character in the books, and there are 102. These include letters, digits, punctuation and rare ones like `é` and `£`.
2. It sorts them, so the same text always gives the same numbers.
3. It numbers them from 0, like a codebook:

```
newline=0   space=1   ,=7   .=9   E=27   H=30   a=52   e=56   h=59   l=63   o=66
```

It has two main functions:
- **`encode`** turns text into numbers: `"hello"` → `[59, 56, 63, 63, 66]`
- **`decode`** turns numbers back into text: `[59, 56, 63, 63, 66]` → `"hello"`

Three things to notice:
- Both `l`s get 63, because the same character always gets the same number.
- `h` is 59 but `H` is 30, because capitals count as different characters.
- An emoji causes an error, because it never appears in the books and so has no number.

A **token** is one piece of text that gets a number. Here each piece is one character, which is why it's called a character-level tokenizer.

## Step 3: Keep some text hidden
[data.py:36-38](../data.py#L36-L38) cuts the numbers into two parts:

| Part | Size | Purpose |
|---|---|---|
| **Training** (first 90%) | 2,368,000 characters | The model studies this |
| **Validation** (last 10%) | 263,112 characters | The model never studies this, so we can test it |

Think of a student who sees the exam questions in advance: a high score could just mean they memorised the answers. Testing on hidden text shows whether the model really learned the patterns.

## Step 4: Make the flashcards
This is the most important step. The model reads small pieces of text, not the whole book at once. [get_batch](../data.py#L42-L61) picks a random spot and takes two pieces of the same length, where the second starts one character later. This is a real example from your data:

```
x (input):   here to dr
y (answers): ere to dra
```

Reading them position by position gives one question per character:

```
after "h"           comes "e"
after "he"          comes "r"
after "her"         comes "e"
after "here"        comes " "
...
after "here to dr"  comes "a"
```

So this 10-character piece holds 10 questions. Your real settings in [config.py](../config.py) are bigger:
- **`BLOCK_SIZE = 128`:** each piece is 128 characters long.
- **`BATCH_SIZE = 32`:** the model gets 32 pieces at a time, called a **batch**.

That makes 32 × 128 = **4,096 questions in every batch**. The batch is stored as a table of numbers with 32 rows and 128 columns, which PyTorch calls a **tensor**.

The spots are random so the model doesn't just go through the book in the same order every time. The code also never starts too close to the end, because the last character still needs a next character as its answer.

## Step 5: Settings and hardware
[config.py](../config.py) keeps all the settings in one place, so you only change them there. Two more settings are in this file:
- **`DEVICE`:** which chip does the maths. Yours is `mps`, your Mac's graphics chip, which is much faster for this work than the main processor.
- **`SEED`:** makes the "random" picks repeat exactly, so you can compare runs fairly.

## Step 6: Tests
[test_stage1.py](../tests/test_stage1.py) has five automatic checks:

| Check | What it proves |
|---|---|
| Text → numbers → text | You get back exactly the original text |
| Numbers are valid | Every number is within the codebook |
| Codebook is sorted, no repeats | `"banana"` gives `a, b, n` |
| Batch shape and shift | The answers really are one character ahead |
| 90/10 split | The split is the right size and no text is lost |

The shift check matters most. If the answers were off by one, the model would learn the wrong thing and you might not notice for a long time.

To see all of this running, use `.venv/bin/python tests/test_stage1.py`.

## Check yourself
If you can answer these without looking, you're ready for Stage 2:

1. Why do we turn characters into numbers?
2. Why do `e` and `E` get different numbers?
3. Why do we hide 10% of the text from the model?
4. If `x` is `Watson` and the next character in the book is `!`, what is `y`? How many questions is that?
5. What do 32 and 128 mean, and how many questions are in one batch?
6. Has the model learned anything after Stage 1? Why or why not?

Send me your answers in your own words and I'll tell you what's right and clear up anything that's still confusing.

**If you read your notes:** [01-tokenization.md](../internal-docs/ai/01-tokenization.md) shows `a` as 40, but the real number is 52. It also says about 2.7 million characters, but the real count is about 2.6 million.
