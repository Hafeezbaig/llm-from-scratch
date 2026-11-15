"""Central configuration for the whole project.

Every other module imports its settings from here, so there is exactly one place
to change them. Only the settings that are actually used so far live here
(Stage 1 = data + tokenizer, Stage 2 = the bigram baseline, Stage 3 = the
transformer).
"""

import torch

# --- Data ---------------------------------------------------------------------
DATA_PATH = "data/input.txt"   # the training corpus: one big plain-text file
TRAIN_SPLIT = 0.9              # fraction used for training (the rest = validation)

# --- Batching -----------------------------------------------------------------
BLOCK_SIZE = 128   # context length: how many characters the model sees at once
BATCH_SIZE = 32    # how many independent sequences we process in parallel

# --- Model --------------------------------------------------------------------
N_EMBD = 128   # length of the vector that represents each position
N_HEAD = 4     # attention heads per block (N_EMBD must divide evenly by this)
N_LAYER = 4    # how many transformer blocks are stacked

# --- Training -----------------------------------------------------------------
LEARNING_RATE = 1e-3   # how big a step the optimizer takes on each update
MAX_ITERS = 5000       # how many training steps to run
EVAL_INTERVAL = 500    # measure train/val loss every this many steps
EVAL_ITERS = 200       # batches averaged per loss measurement (less noisy)

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
