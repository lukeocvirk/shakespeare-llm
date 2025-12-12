import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.up = nn.Linear(in_features=dim, out_features=dim * expansion)
        self.down = nn.Linear(in_features=dim * expansion, out_features=dim)
        self.down._is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=False)
        self.output = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.output._is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)

        q, k, v = rearrange(
            qkv,
            "b t (three h d) -> three b h t d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        attn = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        attn = attn / math.sqrt(self.head_dim)
        seq_len = attn.size(-1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.output(out)

        return out
    
class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int,
        num_heads: int,
    ):
        super().__init__()
        self.dim = dim
        self.expansion = expansion
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CausalMultiHeadSelfAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MultiLayerPerceptron(dim, expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
    
class LanguageModel(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([AttentionBlock(dim, expansion, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.unembed = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.unembed(x)

        return x

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # Residual projections use a reduced std to keep variance stable across layers.
            std = 0.02 / math.sqrt(2 * self.num_layers) if getattr(module, "_is_residual", False) else 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
