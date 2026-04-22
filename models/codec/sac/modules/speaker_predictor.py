import torch
import torch.nn as nn

class SpeakerPredictor(nn.Module):
    """Speaker embedding projection module."""

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 256,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        fuse: bool = True,
        use_mean_std: bool = True,
        predict_from: str = "prenet", # "prenet" | "prenet_in" | "acoustic" | "acoustic_q"
        **kwargs,
    ):
        super().__init__()

        self.use_mean_std = use_mean_std
        self.fuse = fuse
        self.predict_from = predict_from

        if self.fuse:
            # fuse=True: input is 2D [B, D], use Linear layers.
            proj_in = input_dim * (2 if use_mean_std else 1)
            self.proj = nn.Sequential(
                nn.Linear(proj_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            # fuse=False: input is 3D [B, D, T], use Conv1d to keep time dimension.
            self.proj = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=1),  # 1x1 conv is per-time-step linear projection.
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape [B, D, T]
        Returns:
            proj_feat (torch.Tensor): shape [B, output_dim]
        """
        if self.fuse:
            sim_mean = x.mean(dim=2)  # [B, D]
            if self.use_mean_std:
                sim_std = x.std(dim=2)  # [B, D]
                feat = torch.cat([sim_mean, sim_std], dim=-1)
            else:
                feat = sim_mean
        else:
            feat = x

        return self.proj(feat)
