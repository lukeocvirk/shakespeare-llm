import argparse
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from dataset import decode, encode
from model import LanguageModel


def select_device(preferred: Optional[Union[str, torch.device]] = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(
    checkpoint_path: Union[str, Path], device: torch.device
) -> tuple[LanguageModel, Dict[str, int], Dict[int, str], Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    required_keys = ["dim", "expansion", "num_heads", "num_layers"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Checkpoint config missing required keys: {missing}")

    model = LanguageModel(
        dim=config["dim"],
        expansion=config["expansion"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        vocab_size=len(checkpoint["stoi"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint["stoi"], checkpoint["itos"], config


@torch.no_grad()
def sample(
    model: LanguageModel,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    prefix: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
    block_size: Optional[int] = None,
) -> str:
    device = select_device(device)
    model.to(device)

    input_ids = torch.tensor([encode(prefix, stoi)], dtype=torch.long, device=device)
    block_size = block_size or input_ids.size(1)

    for _ in range(max_new_tokens):
        x = input_ids[:, -block_size:]
        logits = model(x)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None and top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_values, torch.tensor(float("-inf"), device=device), logits)

        probs = F.softmax(logits, dim=-1)

        if top_p is not None and 0 < top_p < 1:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs = torch.where(cutoff, torch.tensor(0.0, device=device), sorted_probs)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = torch.gather(sorted_indices, -1, next_token)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return decode(input_ids[0].tolist(), itos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoregressively sample from a trained checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file (epoch_xxx.pt).")
    parser.add_argument("--prefix", type=str, default="ROMEO:", help="Starting text for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling cutoff.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, mps).")
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Context window; defaults to prefix length if not provided.",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    model, stoi, itos, _ = load_checkpoint(args.checkpoint, device=device)

    generated = sample(
        model=model,
        stoi=stoi,
        itos=itos,
        prefix=args.prefix,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        block_size=args.block_size,
    )
    print(generated)


if __name__ == "__main__":
    main()
