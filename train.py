from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from dataset import create_dataloaders
from model import LanguageModel


def train(
    batch_size: int = 64,
    block_size: int = 128,
    dim: int = 256,
    expansion: int = 4,
    num_heads: int = 4,
    num_layers: int = 6,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    num_epochs: int = 5,
    grad_clip: float = 1.0,
    train_fraction: float = 0.9,
    device: Optional[Union[str, torch.device]] = None,
    seed: int = 42,
    checkpoint_dir: Union[str, Path] = "checkpoints",
) -> Tuple[LanguageModel, Dict[str, int], Dict[int, str]]:
    """
    Train the LanguageModel on Tiny Shakespeare with epoch-wise validation and checkpointing.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if device is not None:
        device = torch.device(device)
    else:
        mps_ok = getattr(torch.backends, "mps", None)
        if mps_ok is not None and mps_ok.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    train_loader, val_loader, stoi, itos = create_dataloaders(
        batch_size=batch_size,
        block_size=block_size,
        train_fraction=train_fraction,
    )
    vocab_size = len(stoi)

    model = LanguageModel(
        dim=dim,
        expansion=expansion,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(epoch: int) -> None:
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "config": {
                    "batch_size": batch_size,
                    "block_size": block_size,
                    "dim": dim,
                    "expansion": expansion,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "train_fraction": train_fraction,
                    "seed": seed,
                },
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_total = 0.0

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss_total += loss.item()
            progress = f"Epoch {epoch}/{num_epochs} [{step}/{len(train_loader)}] train loss {loss.item():.4f}"
            print(progress, end="\r")

        avg_train_loss = train_loss_total / len(train_loader)
        print()  # newline after progress bar

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for step, (xb, yb) in enumerate(val_loader, start=1):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), yb.view(-1), reduction="mean"
                )
                val_loss_total += loss.item()
                progress = f"Val   {epoch}/{num_epochs} [{step}/{len(val_loader)}] val loss {loss.item():.4f}"
                print(progress, end="\r")

        avg_val_loss = val_loss_total / len(val_loader)
        print()  # newline after progress bar

        print(
            f"Epoch {epoch}/{num_epochs} completed | train loss {avg_train_loss:.4f} | val loss {avg_val_loss:.4f}"
        )
        save_checkpoint(epoch)

    return model, stoi, itos


if __name__ == "__main__":
    train()
