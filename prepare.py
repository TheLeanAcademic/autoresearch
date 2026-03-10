"""
One-time data preparation for autoresearch experiments (CPU mode).
Downloads TinyStories data shards. Uses a built-in byte-level tokenizer
(no external tokenizer training needed).

Usage:
    python prepare.py                  # full prep (download all shards)
    python prepare.py --num-shards 2   # download only 2 train shards (for testing)

Data is stored in ~/.cache/autoresearch/.
"""

import os
import math
import time
import argparse

import requests
import pyarrow.parquet as pq
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 256        # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 256 * 512  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")

BASE_URL = "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/data"

TRAIN_FILES = [
    "train-00000-of-00004.parquet",
    "train-00001-of-00004.parquet",
    "train-00002-of-00004.parquet",
    "train-00003-of-00004.parquet",
]
VAL_FILES = ["val-00000-of-00001.parquet"]
ALL_FILES = TRAIN_FILES + VAL_FILES

VOCAB_SIZE = 256  # byte-level tokenizer: tokens are raw UTF-8 bytes (0–255)
BOS_TOKEN_ID = 0  # repurpose byte 0x00 as BOS (0x00 never appears in valid UTF-8 text)

# Device selection: prefer MPS then CUDA, fall back to CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _download_file(filename):
    """Download one parquet file with retries. Returns True on success."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_train_shards=len(TRAIN_FILES)):
    """Download training shards + validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    files_to_download = TRAIN_FILES[:num_train_shards] + VAL_FILES

    existing = sum(1 for f in files_to_download if os.path.exists(os.path.join(DATA_DIR, f)))
    if existing == len(files_to_download):
        print(f"Data: all {len(files_to_download)} files already downloaded at {DATA_DIR}")
        return

    needed = len(files_to_download) - existing
    print(f"Data: downloading {needed} files ({existing} already exist)...")

    results = [_download_file(f) for f in files_to_download]
    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(files_to_download)} files ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Byte-level tokenizer (no training needed)
# ---------------------------------------------------------------------------

class Tokenizer:
    """
    Byte-level tokenizer: each UTF-8 byte maps to token id 0–255.
    BOS token is id 0 (byte 0x00, repurposed — rarely appears in natural text).
    """

    def get_vocab_size(self):
        return VOCAB_SIZE

    def get_bos_token_id(self):
        return BOS_TOKEN_ID

    def encode(self, text, prepend=None):
        """Encode a string or list of strings to byte token ids."""
        if isinstance(text, str):
            ids = list(text.encode("utf-8"))
            if prepend is not None:
                ids.insert(0, prepend)
            return ids
        elif isinstance(text, list):
            result = []
            for t in text:
                ids = list(t.encode("utf-8"))
                if prepend is not None:
                    ids.insert(0, prepend)
                result.append(ids)
            return result
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    @classmethod
    def from_directory(cls, tokenizer_dir=None):
        """For compatibility with train.py interface — no file loading needed."""
        return cls()


def get_token_bytes(device="cpu"):
    """
    Returns a tensor of byte counts per token id (all 1 for byte-level tokens).
    Token 0 (BOS) is assigned byte length 0 so it is excluded from BPB calculation.
    """
    token_bytes = torch.ones(VOCAB_SIZE, dtype=torch.int32, device=device)
    token_bytes[BOS_TOKEN_ID] = 0  # BOS: excluded from BPB
    return token_bytes


# ---------------------------------------------------------------------------
# Data loading utilities (imported by train.py)
# ---------------------------------------------------------------------------

def _document_batches(split, batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    if split == "train":
        filenames = [f for f in ALL_FILES if f.startswith("train-")]
    else:
        filenames = VAL_FILES
    filepaths = [os.path.join(DATA_DIR, f) for f in filenames]
    missing = [p for p in filepaths if not os.path.exists(p)]
    assert not missing, f"Missing data files: {missing}. Run prepare.py first."
    epoch = 1
    while True:
        for filepath in filepaths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), batch_size):
                    yield batch[i : i + batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding). Yields CPU tensors.
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    inputs = torch.empty((B, T), dtype=torch.long)
    targets = torch.empty((B, T), dtype=torch.long)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        inputs.copy_(row_buffer[:, :-1])
        targets.copy_(row_buffer[:, 1:])
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. BOS tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device=DEVICE)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    # max(1,...) ensures at least one eval step even if batch_size*MAX_SEQ_LEN > EVAL_TOKENS
    steps = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TinyStories data for autoresearch (CPU mode)")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=len(TRAIN_FILES),
        help=f"Number of training shards to download (1–{len(TRAIN_FILES)}). Val shard is always downloaded.",
    )
    args = parser.parse_args()

    num_train = max(1, min(args.num_shards, len(TRAIN_FILES)))

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Device:          {DEVICE}")
    print()

    download_data(num_train_shards=num_train)
    print()
    print("Done! Ready to train.")
    print(f"  Tokenizer: byte-level (vocab_size={VOCAB_SIZE}, no training needed)")
    print(f"  MAX_SEQ_LEN={MAX_SEQ_LEN}, EVAL_TOKENS={EVAL_TOKENS}, TIME_BUDGET={TIME_BUDGET}s")
