from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DEFAULT_CACHE_DIR = Path("data")
DEFAULT_CACHE_FILE = "tiny_shakespeare.txt"


def download_shakespeare(
    url: str = DEFAULT_DATA_URL,
    cache_dir: Union[Path, str] = DEFAULT_CACHE_DIR,
    cache_file: str = DEFAULT_CACHE_FILE,
    refresh: bool = False,
) -> str:
    """
    Download the Tiny Shakespeare dataset or read it from a local cache.
    """
    cache_path = Path(cache_dir) / cache_file
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download dataset (status {response.status_code}).")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text]


def decode(tokens: Iterable[int], itos: Dict[int, str]) -> str:
    return "".join(itos[int(tok)] for tok in tokens)


def split_data(encoded: np.ndarray, train_fraction: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    cutoff = int(train_fraction * len(encoded))
    return encoded[:cutoff], encoded[cutoff:]


class ShakespeareDataset(Dataset):
    def __init__(self, encoded_data: np.ndarray, block_size: int):
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if len(encoded_data) <= block_size:
            raise ValueError("encoded_data must be longer than block_size.")

        self.data = torch.tensor(encoded_data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError("Data length must exceed block_size.")

    idx = np.random.randint(0, len(data) - block_size, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in idx])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in idx])
    x_tensor = torch.tensor(x, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)

    if device is not None:
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

    return x_tensor, y_tensor


def load_shakespeare(
    url: str = DEFAULT_DATA_URL,
    cache_dir: Union[Path, str] = DEFAULT_CACHE_DIR,
    train_fraction: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Download (or load cached) data, build the vocabulary, and return train/val splits.
    """
    text = download_shakespeare(url=url, cache_dir=cache_dir)
    stoi, itos = build_vocab(text)

    encoded = np.array(encode(text, stoi), dtype=np.int64)
    train_data, val_data = split_data(encoded, train_fraction=train_fraction)

    return train_data, val_data, stoi, itos


def create_dataloaders(
    batch_size: int,
    block_size: int,
    url: str = DEFAULT_DATA_URL,
    cache_dir: Union[Path, str] = DEFAULT_CACHE_DIR,
    train_fraction: float = 0.9,
    num_workers: int = 0,
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """
    Convenience helper to build PyTorch dataloaders and vocab metadata in one call.
    """
    train_data, val_data, stoi, itos = load_shakespeare(
        url=url, cache_dir=cache_dir, train_fraction=train_fraction
    )

    train_ds = ShakespeareDataset(train_data, block_size=block_size)
    val_ds = ShakespeareDataset(val_data, block_size=block_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers
    )

    return train_loader, val_loader, stoi, itos
