import json
import torch
import torch.nn as nn
from models.codec.sac.modules.utils.ERes2Net import ERes2Net
from models.codec.sac.modules.utils.fbank import FBank

ERes2Net_VOX = {
    "args": {"feat_dim": 80, "embedding_size": 192},
}

class SpeakerEmbedder(nn.Module):
    """Load pretrained speaker model and extract embeddings from waveform tensor."""

    def __init__(self,
                 pretrained_dir: str,
                 freeze: bool = True,
                 ):
        super().__init__()
        
        with open(f"{pretrained_dir}/configuration.json", "r") as f:
            config = json.load(f)
        pretrained_model = config["model"]["pretrained_model"]
        self.model = ERes2Net(**ERes2Net_VOX["args"])
        state = torch.load(f"{pretrained_dir}/{pretrained_model}", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)

        self.feat_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        self.embedding_dim = ERes2Net_VOX["args"]["embedding_size"]

        self.predict_from = "wav"

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            
            for param in self.model.pool.parameters():
                param.requires_grad = True
            for param in self.model.seg_1.parameters():
                param.requires_grad = True
            for param in self.model.seg_2.parameters():
                param.requires_grad = True
            
            # if len(self.model.layer4) > 0:
            #     for param in self.model.layer4[-1].parameters():
            #         param.requires_grad = True
            
            self.model.train()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav (torch.Tensor): [B, T] waveform, already 16kHz mono
        Returns:
            embeddings: [B, embedding_dim]
        """
        if wav.ndim == 1:  # single example
            wav = wav.unsqueeze(0)
        feats = [self.feat_extractor(w) for w in wav]       # list of [T_f, 80]
        feats = torch.stack(feats)                          # [B, T_f, 80]
        # emb = self.model(feats).detach()                    # [B, D_emb]
        emb, latent = self.model(feats) 
        if not self.model.training:
            emb = emb.detach()
            latent = latent.detach()
        return emb, latent
