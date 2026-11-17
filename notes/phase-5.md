# Stage 5, explained simply

## Where we left off
Robot 2 finished studying in Stage 4, with a loss of **1.08**, and its brain is saved in `checkpoints/gpt.pt`. To write, it does the same thing for every character: it gives each of the 102 characters a score, turns the scores into odds, and **rolls a weighted dice**.

Until now we always rolled the dice exactly as the robot made it. Stage 5 adds **two knobs** to the dice:

1. a **boldness knob** (temperature)
2. a **shortlist knob** (top-k)

There's **no new training**: it's the same brain, only a different way of picking. That's why this stage runs happily on your Mac. Writing 500 characters takes a few seconds.

## Chapter 1: The weighted dice (a reminder)
Give the robot `the game is a` and ask: what comes next? Here are its odds (from `python generate.py --peek "the game is a" --temperature 1.0`):

```
    space   60.8%  ████████████████████████
        n    9.5%  ████
        l    8.6%  ███
  newline    7.3%  ███
        s    4.4%  ██
        b    3.1%  █
        t    2.8%  █
        w    0.6%
```

The robot thinks a space is most likely ("the game is a **long** one"?). But a space isn't certain, so a roll can still give `n` ("an…") or `l`. Every roll adds one character, then the robot looks again and rolls again.

> In the code: `generate()` in [model.py](../model.py): scores → `next_char_probs()` → `torch.multinomial` (the dice roll).

## Chapter 2: The boldness knob (temperature)
Think of two writers. A **shy** writer almost always picks the safest word. A **daring** writer often picks a surprising one. Temperature turns the robot into either.

**How it works:** before the scores become odds, every score is **divided by the temperature**. That changes how far apart the scores are:

| Two characters with scores 4 and 2 | Gap | The favourite's odds |
|---|---|---|
| temperature 1 (scores 4 and 2) | 2 | 88% |
| temperature 0.5 (scores become 8 and 4) | 4 | **98%**: the shy writer |
| temperature 2 (scores become 2 and 1) | 1 | **73%**: the daring writer |

- **Below 1:** bigger gaps, so the favourite wins almost every time. **Careful.**
- **Above 1:** smaller gaps, so long shots get a real chance. **Bold.**
- **Exactly 1:** the robot's own odds, which is what Stages 2–4 used.
- **The order never changes.** The favourite stays the favourite; only how much it's favoured changes.

The real robot, after `the game is a`:

| | space | n | l | newline | the other 98 characters |
|---|---|---|---|---|---|
| temperature 0.5 | 93.5% | 2.3% | 1.9% | 1.3% | 1.0% |
| temperature 1.0 | 60.8% | 9.5% | 8.6% | 7.3% | 13.8% |
| temperature 2.0 | 26.4% | 10.5% | 9.9% | 9.1% | **44.1%** |

Look at the last column. At temperature 2.0, almost half of the dice is spread over the 98 unlikely characters, including junk like `º` and `X`. That's where nonsense comes from.

**Temperature 0 isn't allowed,** because you can't divide by zero. To always pick the favourite, use the other knob: `top_k=1`.

> In the code: `logits = logits / temperature` in `next_char_probs()` in [model.py](../model.py).

## Chapter 3: The shortlist knob (top-k)
Imagine a quiz show where only the **top k answers** are allowed on the board. Everyone else's answer is crossed out, and the survivors share 100% of the chances.

With `top_k=3`, after `the game is a`:

```
    space   77.1%  ███████████████████████████████
        n   12.1%  █████
        l   10.8%  ████
```

Newline, `s`, `b` and the other 96 characters now have **exactly 0%**, so they can never be picked.

**Why it helps:** the shortlist throws out the 98 long shots, which is exactly where temperature 2.0 found its junk. So you can be bold **within a safe list**.

- `top_k=1` = only the favourite, every time. The dice stop mattering.
- `top_k` bigger than the alphabet (over 102) = nothing changes.

> In the code: `torch.topk` and `masked_fill(..., -inf)` in `next_char_probs()`. A score of minus infinity becomes a probability of exactly 0.

## Chapter 4: Trying every setting
`python generate.py --compare` writes from the same start with seven settings. For each setting it writes 4 pieces of 500 characters and gives them two scores:

- **Real words:** how many of the words also appear somewhere in the books. Made-up words like "exhaustrated" don't count.
- **Variety:** different words ÷ all words. Low means the text keeps repeating itself.

| Setting | Temperature | Top-k | Real words | Variety |
|---|---|---|---|---|
| Always the favourite | 1.0 | 1 | 100% | 28% |
| Careful | 0.5 | off | 99% | 57% |
| **A bit careful (the default)** | **0.8** | **off** | **99%** | **67%** |
| The robot's own odds | 1.0 | off | 95% | 75% |
| Bold | 1.5 | off | 79% | 87% |
| Wild | 2.0 | off | 52% | 93% |
| Wild + shortlist of 5 | 2.0 | 5 | 89% | 76% |

What each one actually wrote:

**Always the favourite** gets stuck in a circle:
```
“I have no doubt that I have no doubt that I have no doubt that
the matter had been so much as the man who had been seen the
strange that the strange was a strange constable and the street
```
Why? With no dice, the same text in view always leads to the same next character. Once it writes "I have no doubt that", the favourite next words are "I have no doubt that" again, and round it goes.

**Careful (0.5):** tidy and real, but a bit flat:
```
The house was always a strange brown story, and the evening had been str
```

**A bit careful (0.8)**, the default in `generate.py`:
```
On the Duke of the City has been formed to me again, and that the hansom
had been a little prospect at the provision of the station.
```

**The robot's own odds (1.0):** livelier, with a few made-up words ("exhaustrated", "duplicatear"):
```
a succeeded house and bright like a duplicatear below, blue high thrush,
```

**Bold (1.5):** the cracks show:
```
At, quite among my twenty-skirts, whence EvalUs posts Coombe Straplain.
```

**Wild (2.0):** falls apart:
```
VPaYkºrl’s Sherlock; and poson ChooseleN had iunbaldties.”
```

**Wild + shortlist of 5:** the same boldness, but the junk is gone. Real words climb from 52% back to 89%:
```
He stooped over along its plaim and r
```

**The big lesson:** no setting wins both scores. Careful = makes sense but repeats. Bold = fresh but messy. You choose the balance, and that's what "creativity vs coherence" means.

## Chapter 5: Peeking at the dice, and the "afoot" puzzle
In Stage 4, the robot didn't finish "the game is a" with `f` (for "afoot"). With `--peek`, we can see exactly how likely `f` is:

| | Chance of `f` | Its place in the list |
|---|---|---|
| temperature 0.5 | 0.01% | 9th |
| temperature 1.0 | 0.51% | 9th |
| temperature 2.0 | 2.42% | 9th |

- Boldness makes the rare-but-right `f` almost **5 times more likely**. But it also makes every piece of junk more likely. That's the deal temperature offers.
- `f` is 9th, so a shortlist of 8 (`top_k=8`) would **rule it out completely**. Both knobs have a cost.

## Chapter 6: The helpers in generate.py
- **Loading the saved brain:** `load_model()` reads the "photo" from Stage 4 (the numbers, the settings and the alphabet) and rebuilds the robot. No training needed.
- **Curly quotes:** the books use “ ” but your keyboard types `"`. So `generate.py` swaps them for you: `"Come, Watson` becomes `“Come, Watson`.
- **Unknown characters:** the robot only knows its 102 characters. Type `@` and it explains, rather than crashing:
  `error: the model never saw these characters: ['@']`
- **The seed:** the dice are "random", but they start from a number called the seed. **Same seed + same settings = the same text**, every time. We checked: two runs gave identical text. Change `--seed` to get a different piece.

> Fun fact: the AI services that big apps are built on often have a setting called **temperature**. It's this same knob, on a much bigger dice.

## The end of Stage 5
The robot now writes on demand, from any start you give it, as carefully or as boldly as you like. Next, in Stage 6, we tidy the project up, run **one real experiment** with a graph (for example, a small brain against a big one), and write it all up.

## The five words to remember
**Scores → odds → divide → shortlist → seed**

And one rule: *careful makes sense but repeats; bold is fresh but messy.*

## Story → code
| Story | Code |
|---|---|
| The weighted dice | `torch.multinomial(probs, 1)` in `generate()` in [model.py](../model.py) |
| The boldness knob | `temperature`: `logits / temperature` in `next_char_probs()` |
| The quiz-show shortlist | `top_k`: `torch.topk` + `masked_fill(-inf)` in `next_char_probs()` |
| Loading the saved brain | `load_model()` in [generate.py](../generate.py) |
| Curly quotes and unknown characters | `prepare_prompt()` |
| Peeking at the dice | `peek()`, `python generate.py --peek "…"` |
| Trying every setting | `print_compare()`, `python generate.py --compare` |
| The two scores | `real_word_share()` and `variety()` |
| Same text every time | `--seed` → `torch.manual_seed()` |

## Check yourself
1. Does Stage 5 change the robot's brain? Why could it run on your Mac?
2. In your own words: what happens to the scores at temperature 0.5, and at 2.0?
3. Why does the favourite stay the favourite at any temperature?
4. Why is temperature 0 not allowed, and what do you use instead?
5. With `top_k=3`, what's the chance of the 4th-likeliest character?
6. Why did "always the favourite" get stuck saying "I have no doubt that"?
7. Temperature 2.0 gave 52% real words, but adding a shortlist of 5 gave 89%. Why?
8. Why would `top_k=8` make "afoot" impossible after "the game is a"?

## How to make it stick
1. Read one chapter, look away, and say it out loud in one sentence.
2. The next day, write the five words from memory, then check.
3. Run `python generate.py --peek "the game is a"` three times, with `--temperature 0.5`, then `1.0`, then `2.0`. Watch the space's bar shrink.
4. Try your own start: `python generate.py --prompt "Sherlock Holmes" --temperature 0.5`, then the same with `--temperature 1.5`. Which would you rather read?
5. Run `.venv/bin/python tests/test_stage5.py`. Then read `test_temperature_sharpens_or_flattens` in [tests/test_stage5.py](../tests/test_stage5.py): its asserts say "the favourite gets likelier, the long shot less likely, and the order never changes".
6. **Experiment (guess first, then run):** what do you think `python generate.py --temperature 1.5 --top_k 1` writes? Hint: which knob wins when there's only one character on the list?
