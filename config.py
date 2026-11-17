"""Central configuration for the whole project.

Every other module imports its settings from here, so there is exactly one place
to change them. Only the settings that are actually used so far live here
(Stage 1 = data + tokenizer, Stage 2 = the bigram baseline, Stage 3 = the
transformer, Stage 4 = scaling it up and training it properly).
"""

import torch

# --- Data ---------------------------------------------------------------------
DATA_PATH = "data/input.txt"   # the training corpus: one big plain-text file
TRAIN_SPLIT = 0.9              # fraction used for training (the rest = validation)

# --- Batching -----------------------------------------------------------------
BLOCK_SIZE = 128   # context length: how many characters the model sees at once
BATCH_SIZE = 32    # how many independent sequences we process in parallel

# --- Model --------------------------------------------------------------------
# Stage 4 sizes (~10.8M parameters). Stage 3 used N_EMBD=128, N_HEAD=4,
# N_LAYER=4 and no dropout (~0.8M parameters).
N_EMBD = 384   # length of the vector that represents each position
N_HEAD = 6     # attention heads per block (N_EMBD must divide evenly by this)
N_LAYER = 6    # how many transformer blocks are stacked
DROPOUT = 0.2  # fraction of numbers randomly zeroed while training (anti-memorising)

# --- Training -----------------------------------------------------------------
# The step size warms up over WARMUP_ITERS steps, then glides down from
# LEARNING_RATE to MIN_LR by the last step. (Stage 3 used a constant 1e-3 for
# 5,000 steps.)
LEARNING_RATE = 1e-3   # biggest step size, reached after the warm-up
MIN_LR = 1e-4          # step size at the very end of training
WARMUP_ITERS = 100     # steps spent ramping up from ~0 to LEARNING_RATE
MAX_ITERS = 6000       # how many training steps to run
EVAL_INTERVAL = 500    # measure train/val loss every this many steps
EVAL_ITERS = 200       # batches averaged per loss measurement (less noisy)

# --- Outputs ------------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/gpt.pt"    # best model so far (gitignored)
LOSS_PLOT_PATH = "assets/loss_curve.png"  # training graph shown in the README

# --- Bigram baseline training -------------------------------------------------
# A single lookup table can take much bigger steps, and stops improving sooner.
BIGRAM_LEARNING_RATE = 1e-2
BIGRAM_MAX_ITERS = 3000
BIGRAM_EVAL_INTERVAL = 300

# --- Reproducibility ----------------------------------------------------------
SEED = 1337        # fixed seed so separate runs are comparable


# --- Hardware -----------------------------------------------------------------
def get_device() -> str:
    """Pick the best available device, once, for the whole project.

    Prefers Apple-Silicon's GPU (MPS), then CUDA, then falls back to CPU. Using
    this everywhere means the same code runs on a Mac, a Colab GPU, or plain CPU.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()
