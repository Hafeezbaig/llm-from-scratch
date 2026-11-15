# Stage 3, explained simply

## Where we left off
Robot 1 (Stage 2) had a **one-letter memory**. It could only see the character in front of it, so it got stuck at a loss of 2.38.

Now Robot 2 arrives. It can look back at **up to 128 earlier characters** before it guesses. This kind of robot is called a **transformer**, and it's the "T" in GPT.

This is the difference, using real guesses from both robots:

```
Text so far: "Sherlock Holme"      What comes next?

Robot 1 (sees only "e"):    space 31%    r 13%    d 8%
Robot 2 (sees all of it):   s 100%
```

Robot 1 only sees an `e`, and lots of things can follow an `e`. Robot 2 sees "Sherlock Holme" and knows it has to be `s`. The rest of these notes explain how it does that.

## Chapter 1: A badge and a seat number
Picture the text as a row of students sitting in seats, one character per seat. Every student has the same job: **guess the character in the next seat.**

Each student starts with two things:
- **A badge** that says *which* character they are. It's a list of 128 numbers, learned during training.
- **A seat number** that says *where* they sit (seat 0, 1, 2 … up to 127). Each seat also has its own 128 learned numbers.

The badge and the seat numbers are added together, so each student starts with 128 numbers that mean "I'm an `e`, sitting in seat 13."

Why does the seat matter? Without it, the robot would struggle to tell `top` from `pot`: same letters, different order.

> In the code: `token_embedding_table` (the badge) and `position_embedding_table` (the seat) in [model.py](../model.py). The size, 128, is `N_EMBD` in [config.py](../config.py).

## Chapter 2: Asking the classmates (attention)
This is the big new idea. Before guessing, each student can **ask the students sitting before them** for help.

Every student makes three things from their 128 numbers:
- **A question** (query): "What am I looking for?"
- **A label** (key): "What do I have?"
- **An answer** (value): "What I'll tell you if you listen to me."

Each student compares their question with every earlier student's label. A good match means they listen more to that student, and a poor match means they listen less. The listening is shared out so it adds up to 100%. Then the student mixes together everyone's answers, mostly from the ones they listened to most.

That's how the student at the last `e` of "Sherlock Holme" can look back at "Sherlock Holm" and work out that `s` comes next. Nobody tells it what to look for. Training shapes the questions, labels and answers.

### The no-peeking rule
The student in seat 3 is trying to guess what's in seat 4. If it could ask seat 4, it would just copy the answer. So **each student may only listen to themselves and the seats before them**:

```
                listens to →   S   h   e   r
student at "S"  (seat 0):      ✓   ✗   ✗   ✗
student at "h"  (seat 1):      ✓   ✓   ✗   ✗
student at "e"  (seat 2):      ✓   ✓   ✓   ✗
student at "r"  (seat 3):      ✓   ✓   ✓   ✓
```

That triangle of ticks is called the **mask**.

> In the code: `Head` in [model.py](../model.py). Its three layers are `query`, `key` and `value`. The triangle is `tril`, and `masked_fill` blocks the ✗ squares.

## Chapter 3: The cheater
What if we break the no-peeking rule? We tried it: the same robot, with the mask removed, trained for 1,000 steps.

```
step     0 | loss 4.79
step   250 | loss 0.23
step   500 | loss 0.03   ← looks like a genius
step  1000 | loss 0.02   ← almost never wrong
```

Then we asked it to write. All it wrote was empty lines, 200 of them, over and over.

Why? While studying, it never learned to *guess*. It learned to *copy from the next seat*. When writing a new character, there is no next seat yet, so it has nothing to copy, and it falls apart.

**Lesson:** a loss that looks too good to be true usually is. That's why the very first Stage 3 tests check that no student can see the future.

> In the code: `test_head_cannot_see_the_future` and `test_gpt_cannot_see_the_future` in [tests/test_stage3.py](../tests/test_stage3.py).

## Chapter 4: Four questions at once (multi-head)
One question isn't enough. A student might want to know "what letter was just before me?" and also "what word am I in the middle of?"

So each student asks **4 questions at the same time**. Each one, called a **head**, has its own question, label and answer. Each head brings back 32 numbers, and the four are joined back into 128. Again, nobody decides what the heads ask. Training does.

> In the code: `MultiHeadAttention` in [model.py](../model.py). The number of heads is `N_HEAD = 4`.

## Chapter 5: Thinking alone (feed-forward)
After listening, each student **thinks on their own** about what they heard, with no more talking. It's a small two-step calculation: 128 numbers → 512 → back to 128.

Attention is the talking, and feed-forward is the thinking.

> In the code: `FeedForward` in [model.py](../model.py).

## Chapter 6: One round, then four rounds (blocks)
**One round = listen, then think.** That round is called a **block**. Two helpers make it work:

- **Keep your notes (residual connection):** students *add* what they learned to their notes instead of erasing them and starting again, so nothing useful gets lost. Without this, a stack of many rounds is very hard to train.
- **Tidy up (layer norm):** before listening and before thinking, each student's numbers are rescaled to a steady size, so they don't drift huge or tiny over many rounds.

Robot 2 does **4 rounds** in a row, and each round builds on the one before. After the last round, each student's 128 numbers become **102 scores**, one for every possible next character. From there it's exactly Stage 2: chances → loss → nudge → dice.

```
characters
   ↓  badge + seat number          (embeddings)
   ↓  round 1: listen → think      (block)
   ↓  round 2: listen → think
   ↓  round 3: listen → think
   ↓  round 4: listen → think
   ↓  tidy up → 102 scores
chances for the next character
```

> In the code: `Block` and `GPT` in [model.py](../model.py). In `Block`, the `x = x + ...` lines are the "keep your notes" part, and `LayerNorm` is the tidy-up. The number of rounds is `N_LAYER = 4`.

## Chapter 7: The report card, and the writing
Robot 2's brain is much bigger: **834,406 numbers**, compared with Robot 1's 10,404. It studied for 5,000 steps, which took about 3 minutes on your Mac.

```
step     0 | train loss 4.80 | val loss 4.80   ← day one
step   500 | train loss 1.89 | val loss 1.81   ← already beats Robot 1's best (2.38)
step  1000 | train loss 1.60 | val loss 1.54
step  3000 | train loss 1.32 | val loss 1.29
step  5000 | train loss 1.24 | val loss 1.22   ← still going down
```

| Robot | Val loss | As good as guessing blindly among… |
|---|---|---|
| Guessing evenly | 4.62 | 102 characters |
| Robot 1, trained | 2.38 | about 11 characters |
| **Robot 2, trained** | **1.22** | **about 3 characters** |

More real guesses:

```
"my dear Wats"   Robot 1: space 31%   t 13%   e 11%     Robot 2: o 97%
"Baker Stre"     Robot 1: space 31%   r 13%   d 8%      Robot 2: e 98%
```

Robot 1 wrote this:

```
 thenf ug ly. menok cat   her owie pler  thed beensund he tf  the amullimetheroore
```

Robot 2 writes this:

```
      “What sholing had drawing it made, and. We open, candle of a platation
      undersonal than must he could have black and her terrible enormations
      from him to her. It cease I could not to assailows that he had
      excited is solding got out of justice. I may peeple.
```

It's a big jump. Most words are real ("candle", "terrible", "justice"), there are quote marks for dialogue, capital letters after full stops, and even the books' indented lines. But the sentences still don't *mean* anything. It has learned how Holmes's writing **looks**, not what it **says**.

There's one more important difference. Robot 1 got **stuck**. Robot 2 was **still improving** when we stopped. So a bigger brain and more studying should help, and that's Stage 4.

(The val loss is slightly lower than the train loss again. That's the same reason as in Stage 2: the drawer part of the books happens to be a little easier.)

## The end of Stage 3
The real robot works and clearly beats the baseline: **1.22 versus 2.38**. Next, in Stage 4, we make it bigger, train it for longer, and draw a graph of the loss going down.

## The five words to remember
**Badge → listen → many heads → think → rounds**

And one rule: *never peek at the next seat.*

## Story → code
| Story | Code |
|---|---|
| Badge (which character) | `token_embedding_table` |
| Seat number (where it sits) | `position_embedding_table` |
| 128 numbers per student | `N_EMBD = 128` |
| 128 seats in a row | `BLOCK_SIZE = 128` |
| Question / label / answer | `query` / `key` / `value` in `Head` |
| How much to listen to each seat | the attention weights `wei` (softmax makes them add to 100%) |
| No-peeking rule | the mask: `tril` and `masked_fill` |
| Four questions at once | `MultiHeadAttention`, `N_HEAD = 4` |
| Thinking alone | `FeedForward` |
| Keep your notes | `x = x + ...` in `Block` (residual connection) |
| Tidy up | `nn.LayerNorm` |
| One round | `Block` |
| Four rounds | `N_LAYER = 4` |
| The whole robot | `GPT` in [model.py](../model.py) |
| Studying | `train()` in [train.py](../train.py), shared with Robot 1 |

## Check yourself
1. After "Sherlock Holme", why did Robot 1 guess a space but Robot 2 guessed `s`?
2. Which two things are added together to make each student's starting 128 numbers?
3. In your own words, what are the question, the label and the answer?
4. Why can't a student listen to the seats after them? What happened to the robot that could?
5. Why use 4 heads instead of 1?
6. What happens in one round? What do "keep your notes" and "tidy up" do?
7. Robot 1 got stuck at 2.38, but Robot 2 was still improving at 1.22. What does that suggest for Stage 4?

## How to make it stick
1. Read one chapter, look away, and say it out loud in one sentence.
2. The next day, before you read anything, write the five words from memory. Then check.
3. Run `.venv/bin/python tests/test_stage3.py`. Then open [tests/test_stage3.py](../tests/test_stage3.py), find `test_head_cannot_see_the_future`, and read its two `assert` lines: the past stays the same, and the future changes.
4. Open [model.py](../model.py) and find the pieces in order: `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPT`.
5. **Experiment:** in [config.py](../config.py), change `N_LAYER` from `4` to `1`. Before running, write down your guess for the final val loss: closer to 2.38 or to 1.22? Then run `.venv/bin/python train.py` (it's faster with one round), compare with your guess, and set it back to `4`.
