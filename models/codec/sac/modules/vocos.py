from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        condition_dim: Optional[int] = None,
        condition_layer: Optional[int] = 1,
        condition_fuse: Optional[str] = "ln",
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.condition_dim = condition_dim
        self.condition_fuse = condition_fuse
        if condition_dim:
            if condition_fuse == "ln":
                self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6, condition_layer=condition_layer)
            elif condition_fuse == "attn ln":
                self.norm = AttnAdaLayerNorm(condition_dim, dim, eps=1e-6)
            elif condition_fuse == "cat":
                self.norm = nn.LayerNorm(dim, eps=1e-6)
                self.condition_proj = nn.Linear(condition_dim, dim)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.condition_dim and self.condition_fuse == "cat":
            assert cond_embedding_id is not None
            cond = self.condition_proj(cond_embedding_id)    
            B, T, _ = x.shape
            cond_tiled = cond.unsqueeze(1).expand(B, T, -1)  # [B, T, condition_dim]
            x = x + cond_tiled
        if self.condition_dim and self.condition_fuse in ["attn ln", "ln"]:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)                
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        condition_dim (int): Dimension of the condition.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(
        self,
        condition_dim: int,
        embedding_dim: int,
        condition_layer: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim

        self.scale_blocks = [nn.Linear(condition_dim, embedding_dim)]
        for i in range(condition_layer - 1):
            self.scale_blocks.append(nn.GELU())
            self.scale_blocks.append(nn.Linear(embedding_dim, embedding_dim))
        self.scale = nn.Sequential(*self.scale_blocks)

        self.shift_blocks = [nn.Linear(condition_dim, embedding_dim)]
        for i in range(condition_layer - 1):
            self.shift_blocks.append(nn.GELU())
            self.shift_blocks.append(nn.Linear(embedding_dim, embedding_dim))
        self.shift = nn.Sequential(*self.shift_blocks)
    
    def init_weights(self):
        for block in self.scale_blocks:
            if isinstance(block, nn.Linear):
                nn.init.trunc_normal_(block.weight, std=0.02)
                nn.init.constant_(block.bias, 0)
        for block in self.shift_blocks:
            if isinstance(block, nn.Linear):
                nn.init.trunc_normal_(block.weight, std=0.02)
                nn.init.constant_(block.bias, 0)
        nn.init.zeros_(self.scale[-1].weight)
        nn.init.ones_(self.scale[-1].bias)
        nn.init.zeros_(self.shift[-1].weight)
        nn.init.zeros_(self.shift[-1].bias)


    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding)
        shift = self.shift(cond_embedding)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class AttnAdaLayerNorm(nn.Module):
    """
    Content-Centric Attention Adaptive Layer Normalization
    - Query: Content features (x) -> drive style query
    - Key/Value: Condition embedding (cond) -> provide style information
    Args:
        condition_dim (int): Dimension of condition embedding (D)
        embedding_dim (int): Dimension of content features (C)
        n_head (int): Number of attention heads (default: 4)
        condition_layer (int): MLP layers for cond projection (default: 1)
        eps (float): LayerNorm epsilon (default: 1e-6)
    """
    def __init__(
        self,
        condition_dim: int,
        embedding_dim: int,
        n_head: int = 4,
        # condition_layer: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim  # Content feature dimension (C)
        self.n_head = n_head
        self.head_dim = embedding_dim // n_head
        assert self.dim % self.n_head == 0, "embedding_dim must be divisible by n_head"

        # 1. Content projection (Query): (B, T, C) -> (B, T, C)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # 2. Condition projection (Key/Value)
        # # Shared MLP for cond pre-processing (optional depth)
        # cond_blocks = [nn.Linear(condition_dim, condition_dim)]
        # for i in range(condition_layer - 1):
        #     cond_blocks.append(nn.GELU())
        #     cond_blocks.append(nn.Linear(condition_dim, condition_dim))
        # self.cond_mlp = nn.Sequential(*cond_blocks)
        
        # Key: (B, D) -> (B, 1, C) (style feature for matching)
        self.k_proj = nn.Linear(condition_dim, embedding_dim)
        # Value: (B, D) -> (B, 1, 2*C) (scale + shift concatenated)
        self.v_proj = nn.Linear(condition_dim, 2 * embedding_dim)
        
        # 3. Output projection for scale/shift refinement
        self.out_proj = nn.Linear(2 * embedding_dim, 2 * embedding_dim)

        # Initialize to standard LayerNorm at start
        self.init_weights()

    def init_weights(self):
        """Initialize weights to ensure initial behavior = standard LayerNorm"""
        # Content query projection
        nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        nn.init.constant_(self.q_proj.bias, 0)
        
        # Condition MLP
        # for m in self.cond_mlp:
        #     if isinstance(m, nn.Linear):
        #         nn.init.trunc_normal_(m.weight, std=0.02)
        #         nn.init.constant_(m.bias, 0)
        
        # Key/Value projection
        nn.init.trunc_normal_(self.k_proj.weight, std=0.02)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.trunc_normal_(self.v_proj.weight, std=0.02)
        nn.init.constant_(self.v_proj.bias, 0)
        
        # Output projection: scale=1, shift=0 initially
        nn.init.zeros_(self.out_proj.weight[:self.dim])       # Scale weight
        nn.init.ones_(self.out_proj.bias[:self.dim])          # Scale bias (init=1)
        nn.init.zeros_(self.out_proj.weight[self.dim:])       # Shift weight
        nn.init.constant_(self.out_proj.bias[self.dim:], 0)   # Shift bias (init=0)

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Content features (B, T, C) -> [Batch, Time, Channel]
            cond_embedding: Condition embedding (B, D) -> [Batch, Condition_dim]
        Returns:
            x_out: Style-adapted features (B, T, C)
        """
        B, T, C = x.shape  # (Batch, Time, Channel)
        
        # Step 1: Basic LayerNorm on content features
        x_norm = nn.functional.layer_norm(x, (C,), eps=self.eps)  # (B, T, C)
        
        # Step 2: Content -> Query (Q) (driven by content)
        q = self.q_proj(x_norm)  # (B, T, C)
        
        # Step 3: Condition -> Key (K) / Value (V)
        # Pre-process condition embedding
        # cond = self.cond_mlp(cond_embedding)  # (B, D) -> (B, D)
        cond = cond_embedding
        # K: Global style feature for matching (B, 1, C)
        k = self.k_proj(cond).unsqueeze(1)    # (B, D) -> (B, 1, C)
        # V: Global style parameters (scale + shift) (B, 1, 2C)
        v = self.v_proj(cond).unsqueeze(1)    # (B, D) -> (B, 1, 2C)
        
        # Step 4: Multi-Head Cross-Attention (Content Q ← Style KV)
        # Reshape for multi-head: (B, n_head, seq_len, head_dim)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.reshape(B, 1, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, 1, head_dim)
        v = v.reshape(B, 1, self.n_head, 2*self.head_dim).transpose(1, 2)# (B, n_head, 1, 2*head_dim)
        
        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim **-0.5)  # (B, n_head, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=-1)                # (B, n_head, T, 1)
        
        # Attention output: (B, n_head, T, 2*head_dim)
        attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, T, 2*C) # (B, T, 2C)
        attn_out = self.out_proj(attn_out)                               # (B, T, 2C)
        
        # Step 5: Split scale/shift and apply
        scale = attn_out[..., :C]  # (B, T, C) -> Frame-wise scale
        shift = attn_out[..., C:]  # (B, T, C) -> Frame-wise shift
        
        # Step 6: Adaptive normalization
        x_out = x_norm * scale + shift  # (B, T, C)
        
        return x_out


class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.

    Args:
        dim (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
            Defaults to (1, 3, 5).
        lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
            Defaults to 0.1.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=self.get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=self.get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=self.get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

        self.gamma = nn.ParameterList(
            [
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        condition_dim: Optional[int] = None,
        condition_layer: Optional[int] = 1,
        condition_fuse: Optional[str] = "ln",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6, condition_layer=condition_layer)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    condition_dim=condition_dim,
                    condition_layer=condition_layer,
                    condition_fuse=condition_fuse,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, AdaLayerNorm):
            return
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert condition is not None
            x = self.norm(x.transpose(1, 2), condition)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, condition)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self,
        input_channels,
        dim,
        num_blocks,
        layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(
            nn.Conv1d(input_channels, dim, kernel_size=3, padding=1)
        )
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[
                ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x