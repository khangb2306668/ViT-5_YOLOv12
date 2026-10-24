"""Spatial ViT-5-Small encoder adapted for detection.

Adapted from Feng Wang et al., ViT-5 (Apache-2.0), revision
b7b0796062f4b51ae00523cb77c7117ae2dff536:
https://github.com/wangf3014/ViT-5/blob/b7b0796062f4b51ae00523cb77c7117ae2dff536/models_vit5.py
https://github.com/wangf3014/ViT-5/blob/b7b0796062f4b51ae00523cb77c7117ae2dff536/rope.py
License copy: licenses/ViT-5-LICENSE.txt.

Changes: expose normalized patch tokens as a map, interpolate patch APE at
runtime, and construct rotary positions on the input device. Module and
parameter names remain compatible with the author's Small checkpoint.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        work = x.float()
        work = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return self.weight * work.to(original_dtype)


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 384, patch_size: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionRotaryEmbedding(nn.Module):
    """2D RoPE with the author's row/column order and coordinate scaling."""

    def __init__(self, dim: int, pt_seq_len: int, theta: float = 10000) -> None:
        super().__init__()
        self.pt_seq_len = pt_seq_len
        self.register_buffer("freqs", 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = math.isqrt(x.shape[1])
        if grid * grid != x.shape[1]:
            raise ValueError("RoPE requires a square token grid")
        positions = torch.arange(grid, device=x.device, dtype=torch.float32)
        positions = positions / grid * self.pt_seq_len
        axis = (positions[:, None] * self.freqs.to(device=x.device)).repeat_interleave(2, dim=-1)
        angles = torch.cat(
            (
                axis[:, None, :].expand(grid, grid, -1),
                axis[None, :, :].expand(grid, grid, -1),
            ),
            dim=-1,
        ).reshape(grid * grid, 1, -1)
        if angles.shape[-1] != x.shape[-1]:
            raise ValueError("RoPE dimension does not match the attention head")
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6, num_registers: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_registers = num_registers
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
        self.rope = VisionRotaryEmbedding(head_dim // 2, 14)
        self.rope_reg = VisionRotaryEmbedding(head_dim // 2, 2, theta=100)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, count, dim = x.shape
        reg_start = count - self.num_registers
        q, k, v = self.qkv(x).reshape(batch, count, 3, self.num_heads, dim // self.num_heads).unbind(2)
        q = self.q_norm(q).to(q.dtype)
        k = self.k_norm(k).to(k.dtype)
        q = torch.cat((q[:, :1], self.rope(q[:, 1:reg_start]), self.rope_reg(q[:, reg_start:])), dim=1)
        k = torch.cat((k[:, :1], self.rope(k[:, 1:reg_start]), self.rope_reg(k[:, reg_start:])), dim=1)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        ).transpose(1, 2).reshape(batch, count, dim)
        return self.proj_drop(self.proj(output))


class Mlp(nn.Module):
    def __init__(self, dim: int = 384, ratio: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(dim * ratio, dim)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = nn.Identity()
        self.norm2 = RMSNorm(dim)
        self.mlp = Mlp(dim)
        self.gamma_1 = nn.Parameter(torch.full((dim,), 1e-4))
        self.gamma_2 = nn.Parameter(torch.full((dim,), 1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        return x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))


class ViT5SpatialEncoder(nn.Module):
    """ViT-5-Small backbone returning normalized spatial patch features."""

    def __init__(self, depth: int = 12) -> None:
        super().__init__()
        self.embed_dim = 384
        self.num_registers = 4
        self.patch_size = 16
        self.patch_embed = PatchEmbed(self.embed_dim, self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, self.num_registers, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 14 * 14, self.embed_dim))
        self.blocks = nn.ModuleList([Block(self.embed_dim, 6) for _ in range(depth)])
        self.norm = RMSNorm(self.embed_dim)

        for token in (self.cls_token, self.reg_token, self.pos_embed):
            nn.init.trunc_normal_(token, std=0.02)
        self.apply(self._init_linear_weights)

    @staticmethod
    def _init_linear_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _interpolate_pos_embed(self, grid: int) -> torch.Tensor:
        if grid == 14:
            return self.pos_embed
        positions = self.pos_embed.reshape(1, 14, 14, self.embed_dim).permute(0, 3, 1, 2)
        positions = F.interpolate(positions.float(), size=(grid, grid), mode="bicubic", align_corners=False)
        return positions.permute(0, 2, 3, 1).reshape(1, grid * grid, self.embed_dim).to(self.pos_embed.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("ViT-5 expects an RGB batch [B,3,H,W]")
        height, width = x.shape[-2:]
        if height != width or height % self.patch_size:
            raise ValueError("ViT-5 input must be square and divisible by 16")
        grid = height // self.patch_size
        patches = self.patch_embed(x) + self._interpolate_pos_embed(grid)
        tokens = torch.cat(
            (self.cls_token.expand(x.shape[0], -1, -1), patches, self.reg_token.expand(x.shape[0], -1, -1)),
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens)
        patches = self.norm(tokens)[:, 1 : 1 + grid * grid]
        return patches.transpose(1, 2).reshape(x.shape[0], self.embed_dim, grid, grid)
