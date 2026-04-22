"""
ein notation:
b - batch
c - channel (feature dimension, D)
t - time sequence
d - hidden dimension (model dim)
"""

from __future__ import annotations
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb


# Convolutional positional embedding module
class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0, "Kernel size must be odd for symmetric padding"
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Apply mask if provided
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        # Permute for Conv1d: B T D -> B D T
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        # Permute back: B D T -> B T D
        out = x.permute(0, 2, 1)

        # Reapply mask to output
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


# Rotary positional embedding utils
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    """Precompute cos/sin values for rotary positional embedding"""
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    """Get positional indices for rotary embedding"""
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# Adaptive LayerNorm with global condition modulation (2*D dim input)
class AdaLayerNormZero(nn.Module):
    """
    Adaptive LayerNorm with scale/shift/gate modulation from global condition
    Input: global condition (condition_dim dim) → project to 6*d for modulation
    """
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.silu = nn.SiLU()
        # Project global condition to modulation parameters
        self.linear = nn.Linear(cond_dim, dim * 6)
        # Zero init for stable initial behavior
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # No elementwise affine (modulation comes from global condition)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, cond):
        """
        Args:
            x: Input tensor (B T d)
            cond: Global condition tensor (B condition_dim)
        Returns:
            x: Normalized and modulated tensor (B T d)
            gate_msa: Gate for attention (B d)
            shift_mlp: Shift for FFN (B d)
            scale_mlp: Scale for FFN (B d)
            gate_mlp: Gate for FFN (B d)
        """
        # Project global condition to modulation parameters
        cond = self.linear(self.silu(cond))  # B condition_dim → B 6d
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond, 6, dim=1)
        
        # Apply adaptive modulation (zero init → initial state = pure LN)
        x = self.norm(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# Final Adaptive LayerNorm (only scale/shift for output)
class AdaLayerNormZeroFinal(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, dim * 2)
        # Zero init for final modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, cond):
        """Final normalization with global condition modulation"""
        cond = self.linear(self.silu(cond))  # B condition_dim → B 2d
        scale, shift = torch.chunk(cond, 2, dim=1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# Standard MultiHeadAttention Processor
class SelfAttentionProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: 'MultiHeadAttention',
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope=None,
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        x = attn.to_out[0](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint MultiHeadAttention Processor (for cross-modal attention)
class JointAttentionProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: 'MultiHeadAttention',
        x: torch.Tensor,
        c: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        rope=None,
        c_rope=None,
    ) -> torch.FloatTensor:
        residual = x
        batch_size = c.shape[0]

        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        query = torch.cat([query, c_query], dim=1)
        key = torch.cat([key, c_key], dim=1)
        value = torch.cat([value, c_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        x = attn.to_out[0](x)
        
        if not attn.context_pre_only:
            c = attn.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x, c


# MultiHeadAttention Module
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        processor: JointAttentionProcessor | SelfAttentionProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        context_dim: Optional[int] = None,
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MultiHeadAttention requires PyTorch 2.0 or higher.")

        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, dim)
        ])

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        rope=None,
        c_rope=None,
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Unified Input Embedding for x (in_channels_x) and cond (in_channels_c)
class AcousticConverterInputEmbedding(nn.Module):
    def __init__(self, in_channels_x, in_channels_c, out_dim):
        super().__init__()
        # Separate linear layers for x and c (different input dimensions)
        self.linear_x = nn.Linear(in_channels_x, out_dim)
        self.linear_cond = nn.Linear(in_channels_c, out_dim)
        
        self.conv_pos_embed_x = ConvPositionEmbedding(out_dim)
        self.conv_pos_embed_cond = ConvPositionEmbedding(out_dim)
        
        self.precompute_max_pos = 1024
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input acoustic feature (B in_channels_x T)
            cond: Conditional feature (B in_channels_c T_cond)
        Returns:
            x_embed: Embedded input (B T out_dim)
            cond_embed: Embedded condition (B T_cond out_dim)
        """
        # Process x (B in_channels_x T → B T in_channels_x → B T out_dim)
        x = x.permute(0, 2, 1)
        x_embed = self.linear_x(x)
        x_embed = self.conv_pos_embed_x(x_embed) + x_embed
        
        # Process cond (B in_channels_c T_cond → B T_cond in_channels_c → B T_cond out_dim)
        cond = cond.permute(0, 2, 1)
        cond_embed = self.linear_cond(cond)
        
        # Add sinusoidal positional embedding to cond
        batch_size, seq_len, _ = cond_embed.shape
        batch_start = torch.zeros((batch_size,), dtype=torch.long, device=cond_embed.device)
        pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
        pos_embed = self.freqs_cis[pos_idx]
        cond_embed = cond_embed + pos_embed
        
        cond_embed = self.conv_pos_embed_cond(cond_embed) + cond_embed
        
        return x_embed, cond_embed


# Converter Block with global condition
class AcousticConverterBlock(nn.Module):
    def __init__(self, dim, cond_dim, heads, dim_head, ff_mult=4, context_pre_only=False):
        super().__init__()
        self.context_pre_only = context_pre_only

        # Adaptive LN with global condition modulation
        # self.attn_norm_c = AdaLayerNormZero(dim, cond_dim)
        self.attn_norm_c = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.attn_norm_x = AdaLayerNormZero(dim, cond_dim)
        
        self.attn = MultiHeadAttention(
            processor=JointAttentionProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            context_dim=dim,
            context_pre_only=context_pre_only,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
            self.ff_c = FeedForward(dim=dim, mult=ff_mult, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
            
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, approximate="tanh")

    def forward(self, acoustic_latent, frame_condition_embed, speaker_condition, mask=None, rope=None, c_rope=None):
        """
        Args:
            acoustic_latent: Main input embedding (B T d)
            frame_condition_embed: Context embedding (B T_cond d)
            speaker_condition: Utterance-level speaker condition tensor (B condition_dim)
            mask: MultiHeadAttention mask (B T)
            rope: Rotary embedding for x
            c_rope: Rotary embedding for c
        """
        # Normalize + modulate with global condition
        # norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(frame_condition_embed, speaker_condition)
        norm_c = self.attn_norm_c(frame_condition_embed)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(acoustic_latent, speaker_condition)

        # Joint attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)

        # Process context with modulation
        if not self.context_pre_only:
            frame_condition_embed = frame_condition_embed + c_attn_output
            norm_c = self.ff_norm_c(frame_condition_embed)
            frame_condition_embed = frame_condition_embed + self.ff_c(norm_c)
        else:
            frame_condition_embed = None

        # Process main input with modulation
        acoustic_latent = acoustic_latent + x_gate_msa[:, None, :] * x_attn_output
        norm_x = self.ff_norm_x(acoustic_latent) * (1 + x_scale_mlp[:, None, :]) + x_shift_mlp[:, None, :]
        acoustic_latent = acoustic_latent + x_gate_mlp[:, None, :] * self.ff_x(norm_x)

        return frame_condition_embed, acoustic_latent


# Acoustic converter with configurable input dimensions and external global condition
class AcousticConverter(nn.Module):
    def __init__(
        self,
        in_channels_x: int,          # Number of channels for input x.
        in_channels_c: int,          # Number of channels for input c.
        condition_dim: int,          # Dimension of global condition.
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        position_agnostic: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.position_agnostic = position_agnostic
        
        # Save input/output dimensions.
        self.in_channels_x = in_channels_x
        self.in_channels_c = in_channels_c
        self.condition_dim = condition_dim
        self.dim = dim

        # Global condition is provided externally via `speaker_condition`.
        # Input embedding (x and c can have different input channel counts).
        self.input_embed = AcousticConverterInputEmbedding(in_channels_x, in_channels_c, dim)

        # Rotary embedding
        self.rotary_embed = RotaryEmbedding(dim_head)

        # Transformer blocks using external global condition.
        self.transformer_blocks = nn.ModuleList(
            [
                AcousticConverterBlock(
                    dim=dim,
                    cond_dim=condition_dim,  # Global condition dimension.
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    context_pre_only=i == depth - 1,
                )
                for i in range(depth)
            ]
        )

        # Final normalization with global condition
        self.norm_out = AdaLayerNormZeroFinal(dim, condition_dim)
        self.proj_out = nn.Linear(dim, in_channels_x)  # Match x input dimension.

    def forward(
        self,
        acoustic_latent: torch.Tensor,                  # B in_channels_x T
        frame_condition: Optional[torch.Tensor] = None, # B in_channels_c T_cond (can have different length)
        speaker_condition: Optional[torch.Tensor] = None,  # B condition_dim
        mask: Optional[torch.Tensor] = None,  # B T
    ) -> torch.Tensor:
        """
        Forward pass with externally provided global condition
        Args:
            acoustic_latent: Main input feature (B in_channels_x T)
            frame_condition: Frame-level conditioning feature (B in_channels_c T_cond)
            speaker_condition: Speaker-level global condition (B condition_dim)
            mask: Attention mask (B T)
        Returns:
            output: Processed output (B in_channels_x T)
        """
        if frame_condition is None or speaker_condition is None:
            raise ValueError("AcousticConverter expects both frame_condition and speaker_condition.")

        # Step 1: Embed x and condition (different input dims / lengths).
        x_embed, cond_embed = self.input_embed(acoustic_latent, frame_condition)  # x: B T d; cond: B T_cond d

        # Step 2: Rotary embedding for different lengths
        x_seq_len = x_embed.shape[1]
        cond_seq_len = cond_embed.shape[1]
        rope = self.rotary_embed.forward_from_seq_len(x_seq_len)
        if self.position_agnostic:
            c_rope = None
        else:
            c_rope = self.rotary_embed.forward_from_seq_len(cond_seq_len)

        # Step 3: Transformer blocks with global condition
        c = cond_embed
        x = x_embed
        for block in self.transformer_blocks:
            c, x = block(x, c, speaker_condition, mask=mask, rope=rope, c_rope=c_rope)

        # Step 4: Final normalization + projection
        x = self.norm_out(x, speaker_condition)
        output = self.proj_out(x)
        
        # Convert back to B in_channels_x T format
        output = output.permute(0, 2, 1)
        
        return output
