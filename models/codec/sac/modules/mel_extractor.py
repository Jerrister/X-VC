import torchaudio.transforms as TT
import torch.nn as nn
import torch

class MelExtractor(nn.Module):
    """Mel extractor (log-Mel spectrogram in dB scale).
    Input:  (batch_size, 1, length) or (batch_size, length)
    Output: (batch_size, n_mels, mel_length), dB-scaled with dynamic-range compression
    """
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        mel_fmin,     
        mel_fmax,
        num_mels: int,
        top_db: float = 80.0,  # Upper bound for dynamic-range compression.
        eps: float = 1e-9,     # Small epsilon for numerical stability.
    ):
        super().__init__()

        self.num_mels = num_mels

        # 1) Base Mel extraction (power=1 for magnitude; convert to dB afterwards).
        self.mel_extractor = TT.MelSpectrogram(
            sample_rate, 
            n_fft, 
            win_length, 
            hop_length, 
            mel_fmin, 
            mel_fmax, 
            n_mels=num_mels, 
            power=1,          # Magnitude spectrogram first, then dB conversion.
            norm="slaney", 
            mel_scale="slaney"
        )

        # 2) Convert magnitude to dB: 20*log10(amplitude).
        self.amplitude_to_db = TT.AmplitudeToDB(
            stype="magnitude",  # power=1 -> magnitude, power=2 -> power
            top_db=top_db,      # Clamp dynamic range to [-top_db, 0] dB.
        )
        
        self.eps = eps  # Prevent log(0)-like instability.

    def forward(self, x: torch.Tensor, *args):
        """encoder forward.
        
        Args:
            x (torch.Tensor): (batch_size, 1, length)
        
        Returns:
            mel (torch.Tensor): (batch_size, n_mels, mel_length), log-Mel in dB.
        """
        # Normalize input shape: (B,1,T) -> (B,T)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.dim() != 2:
            raise ValueError(f"Input must have shape (B,1,T) or (B,T), got {x.shape}")

        # Step 1: Extract linear Mel magnitude spectrogram.
        mel_linear = self.mel_extractor(x)
        
        # Step 2: Add epsilon for better numerical stability.
        mel_linear = mel_linear + self.eps
        
        # Step 3: Convert to log-Mel in dB.
        mel_log = self.amplitude_to_db(mel_linear)

        return mel_log
