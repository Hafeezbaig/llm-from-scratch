# Stage 2, explained simply

## Where we left off
In Stage 1 we got the classroom ready: the **scroll** (the books), the **roll call** (every character has a number), the **locked drawer** (10% hidden for the exam) and the **flashcards** (the answer is always the next character).

Now the first robot arrives. It's a **bad student on purpose**. Its job isn't to write well. Its job is to prove the whole classroom works, and to set a score that the real robot in Stage 3 has to beat.

```
flashcards → robot guesses → guess gets marked → robot fixes itself a little → repeat → robot writes
```

## Chapter 1: The robot with a one-letter memory
This robot can only remember **one character**: the one it's looking at right now. Everything before that is gone.

So if you show it `e`, it has no idea whether the word was "the", "Holmes" or "street". It only thinks: "I see `e`. What usually comes after `e`?"

That's why it's called a **bigram** model. "Bi" means two: it only ever looks at pairs of characters, this one and the next one.

> In the code: `BigramLanguageModel` in [bigram.py](../bigram.py).

## Chapter 2: The notebook
The robot's whole brain is one notebook. It has a page for every character, so 102 pages. On each page it keeps a score for every character that could come next, so 102 scores per page.

102 pages × 102 scores = **10,404 numbers**. That's everything the robot knows.

When the robot arrives, the notebook is full of random scribbles. This is what it writes on day one:

```
KbQàñ.1ATO?qBV7jé[FçîûH.-Eœ3ïAVPM'NºXP0G&,ïZ[SL5E;t‘UW
```

Pure nonsense, because it hasn't studied yet.

> In the code: the notebook is `nn.Embedding(vocab_size, vocab_size)`. Each of the 10,404 numbers is a **parameter**, and the scores are called **logits**.

## Chapter 3: Scores become chances
A score can be any number, like 3.1 or -2.0. To make a guess, the robot turns the scores on a page into **chances** that add up to 100%. A big score becomes a big chance, and a small score becomes a small chance.

These are real pages from the notebook after studying:

```
after "q":   u 98%
after "t":   h 32%   space 24%   o 9%    e 8%
after "H":   o 43%   e 32%       a 10%   i 7%
after ".":   space 65%   newline 16%   ” 16%
```

Nobody told the robot that `u` comes after `q`, or that `t` is often followed by `h` (as in "the"). It worked that out by itself from the flashcards. **That is learning.**

> In the code: `F.softmax` turns scores into chances.

## Chapter 4: Marking the guess (the loss)
After every guess, the teacher looks at the right answer and asks one question: **what chance did the robot give the right answer?**

- A high chance gives a small mark, a **low loss**. That's good.
- A tiny chance gives a big mark, a **high loss**. That's bad.

That mark is called the **loss**. Think of it as "how surprised was the robot by the right answer?" **Lower is better.**

These are the numbers to remember:

| Robot | Loss | As good as guessing blindly among… |
|---|---|---|
| Guessing evenly (every character equal) | 4.62 | 102 characters |
| Our robot on day one (random notebook) | 5.19 | about 180 characters |
| Our robot after studying | 2.42 | about 11 characters |

Why is the day-one robot *worse* than guessing evenly? The random scribbles make it confident about the wrong answers, and being confidently wrong is punished more than being honestly unsure.

> In the code: `F.cross_entropy` works out the loss.

## Chapter 5: Studying (the training loop)
Studying is the same four steps, again and again:

1. **Pick up a stack of flashcards:** 32 cards × 128 characters = 4,096 questions.
2. **Guess** every answer.
3. **Get marked:** that's the loss.
4. **Fix the notebook a little:** for each of the 10,404 numbers, work out whether nudging it up or down would lower the loss. Then nudge each one a tiny bit in the helpful direction.

The robot repeats this **3,000 times**. On your Mac, that takes about 10 seconds.

How big is each nudge? That's the **learning rate**, and ours is 0.01. You can see what happens with other sizes:

| Learning rate | What happened after 3,000 steps |
|---|---|
| 0.0001 (tiny nudges) | Loss only went from 5.19 to 4.72. It barely learned anything. |
| 0.01 (ours) | Loss reached 2.42. |
| 1.0 (huge nudges) | Loss dropped fast, then bounced around 2.48 and never got as low as 2.42. |

> In the code: `train()` in [train.py](../train.py). It moved there in Stage 3, so both robots share it. Step 4 is three lines:
> - `optimizer.zero_grad()` wipes the old nudge directions.
> - `loss.backward()` works out the new nudge direction for every number.
> - `optimizer.step()` does the nudging.

## Chapter 6: The report card
Every 300 steps, the teacher pauses and tests the robot twice. Once on cards from the study pile (**train loss**), and once on cards from the locked drawer, which it has never studied (**val loss**).

This is your real report card:

```
step     0 | train loss 5.19 | val loss 5.20   ← day one
step   300 | train loss 2.72 | val loss 2.69   ← learning fast
step   600 | train loss 2.48 | val loss 2.44
step  3000 | train loss 2.42 | val loss 2.38   ← stuck here
```

There are three things to notice:
- It learns fast at first, then **stops improving around 2.4**. That's not a bug. With a one-letter memory, about 2.4 is the best this robot can ever do. It needs a better brain, not more studying.
- The drawer score matches the study score, so the robot isn't just memorising the study pile. It's too simple to memorise.
- The drawer score is even slightly lower. That's most likely because the drawer is a different part of the books (the end of the last book), which happens to be a little easier to guess. Nothing is wrong.

> In the code: `estimate_loss()` averages 200 stacks to get each score, because one stack on its own gives a jumpy number.

## Chapter 7: The robot writes
To write, the robot does this:

1. It starts with one character, a newline.
2. It opens that character's page and reads the chances.
3. It **rolls a weighted dice**: a character with a 40% chance comes up about 40% of the time.
4. It writes down what came up, turns to that character's page, and repeats.

This is what your robot wrote after studying:

```
 thenf ug ly. menok cat   her owie pler  thed beensund he tf  the amullimetheroore  mand mowan on.
```

Compare it with day one. There are now word-sized chunks with spaces between them, a few real words ("the", "her", "he", "cat"), and full stops followed by a space. It's still nonsense, though. With a one-letter memory, it can't remember which word it's in the middle of.

**Why roll a dice instead of always picking the top choice?** We tried picking the top choice, and the robot got stuck. After a space, its top choice is another space (21%), because the books have many lines that start with a row of spaces. So it wrote a space, then another, then another, forever. Rolling the dice adds variety and gets it out of loops like that.

> In the code: `generate()` in [bigram.py](../bigram.py). The dice roll is `torch.multinomial`.

## The end of Stage 2
The whole loop now works from start to finish:

```
data → model → loss → train → generate
```

And we have a score to beat: **2.38** on the locked drawer. In Stage 3, the real robot gets a memory that can look back at many earlier characters at once. That's called **attention**. If it can't beat 2.38, something is broken.

## The five words to remember
**Notebook → chances → loss → nudge → dice**

Add one fact: *one-letter memory, so it gets stuck at 2.4.*

## Story → code
| Story | Code |
|---|---|
| Notebook | `nn.Embedding` table, 10,404 parameters |
| Scores | logits |
| Scores → chances | `F.softmax` |
| Marking the guess | loss, `F.cross_entropy` |
| Nudge direction | `loss.backward()` |
| Doing the nudge | `optimizer.step()` (AdamW) |
| Nudge size | `BIGRAM_LEARNING_RATE` in [config.py](../config.py) |
| Report card | `estimate_loss()`, train loss and val loss |
| Weighted dice | `torch.multinomial` |

## Check yourself
1. Why is it called a "bigram" model?
2. How many numbers are in the robot's notebook, and why that many?
3. In your own words, what does the loss measure? Is lower or higher better?
4. Why was the day-one robot (5.19) worse than guessing evenly (4.62)?
5. What are the four steps of studying?
6. Why does the loss get stuck around 2.4, and what would fix it?
7. Why does the robot roll a dice instead of always picking the top choice?

## How to make it stick
1. Read one chapter, look away, and say it out loud in one sentence.
2. The next day, before you read anything, write the five words from memory. Then check.
3. Run `.venv/bin/python bigram.py` yourself and watch the report card appear.
4. Open [train.py](../train.py) and find the four study steps inside `train()`.
5. Change `BIGRAM_LEARNING_RATE` in [config.py](../config.py) to `0.0001`, run it again, and see the loss barely move. Then set it back to `1e-2`.
