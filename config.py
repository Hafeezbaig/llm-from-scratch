"""Central configuration for the whole project.

Every other module imports its settings from here, so there is exactly one place
to change them. Only the settings that are actually used so far live here
(Stage 1 = data + tokenizer). Model hyper-parameters such as ``n_layer`` and
``n_head`` are added later, in Stage 3, when the transformer is built.
"""

import torch

# --- Data ---------------------------------------------------------------------
DATA_PATH = "data/input.txt"   # the training corpus: one big plain-text file
TRAIN_SPLIT = 0.9              # fraction used for training (the rest = validation)

# --- Batching -----------------------------------------------------------------
BLOCK_SIZE = 128   # context length: how many characters the model sees at once
BATCH_SIZE = 32    # how many independent sequences we process in parallel

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
