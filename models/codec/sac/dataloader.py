import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.base.base_dataloader import BaseDataset
from omegaconf import DictConfig
from utils.audio import audio_highpass_filter, load_audio

def stack_tensors_with_aligned_T(v: list[torch.Tensor], k: str, pad_value: float = 0.0) -> torch.Tensor:
    """Reuse alignment logic and auto-handle 1D/2D tensors."""
    if not v:
        raise ValueError("Input list `v` must not be empty.")
    
    tensor_dims = set(t.ndim for t in v)
    if len(tensor_dims) != 1:
        raise ValueError(f"All tensors must have the same rank. Found ranks: {tensor_dims}")
    target_dim = tensor_dims.pop()
    
    if target_dim == 2:
        D_list = [t.shape[0] for t in v]
        if len(set(D_list)) != 1:
            raise ValueError(f"For 2D tensors, the first dimension must match. Found: {D_list}")
    
    T_list = [t.shape[-1] for t in v]  # For both 1D/2D tensors, the last dim is treated as T.
    if k in ["source_wav", "target_wav"]:
        T_aligned = 38400
    elif k == "semantic_tokens":
        T_aligned = 30
    elif k == "ssl_feat":
        T_aligned = 120
    # if align_mode == "pad":
    #     T_aligned = max(T_list)
    # elif align_mode == "truncate":
    #     T_aligned = min(T_list)
    # else:
    #     raise ValueError(f"align_mode must be 'pad' or 'truncate', got: {align_mode}")
    
    aligned_tensors = []
    for tensor in v:
        current_T = tensor.shape[-1]
        if current_T < T_aligned:
            pad_num = T_aligned - current_T
            padded = F.pad(
                tensor, 
                pad=(0, pad_num),  # Only pad along the last dimension (T).
                mode="constant", 
                value=pad_value,
            )
            aligned_tensors.append(padded)
        elif current_T > T_aligned:
            if target_dim == 1:
                truncated = tensor[:T_aligned]
            else:
                truncated = tensor[:, :T_aligned]
            aligned_tensors.append(truncated)
        else:
            aligned_tensors.append(tensor)
    
    return torch.stack(aligned_tensors, dim=0)

class VCSSLWAVDataset(BaseDataset):
    """Initialize the dataset with the given configuration.

    Args:
        config (DictConfig):
            Configuration dictionary specifying dataset parameters.
        mode (str):
            Dataset mode, typically 'train' or 'val'.
    """

    def __init__(
        self,
        config: DictConfig,
        mode: str = "train",
        reconstruction_ratio: float = 0.0,
        reversed_ratio: float = 0.0,
        mask_target_condition: bool = False,
        reconstruct_ratio: Optional[float] = None,
        swap_ratio: Optional[float] = None,
        mask_cond: Optional[bool] = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(config, mode)
        # Backward-compatible aliases:
        # reconstruct_ratio -> reconstruction_ratio
        # swap_ratio -> reversed_ratio
        # mask_cond -> mask_target_condition
        self.reconstruction_ratio = reconstruct_ratio if reconstruct_ratio is not None else reconstruction_ratio
        self.reversed_ratio = swap_ratio if swap_ratio is not None else reversed_ratio
        self.mask_target_condition = mask_cond if mask_cond is not None else mask_target_condition
        self.seed = seed
        random.seed(seed)
        
    def fetch_data(self, elem: Dict) -> Dict:
        """Fetch a single sample.

        Args:
            elem (Dict): A dictionary with the key 'index' as index.

        Returns:
            Dict:
        """
        utt = elem["target_utt"][:-3]
        cfg = self.config
        ssl_path = elem['target_semantic_path'] if 'target_semantic_path' in elem else None
        semantic_token_path = elem['source_token_path'] if 'source_token_path' in elem else None
        source_wav_path = elem["source_wav_path"]
        target_wav_path = elem["target_wav_path"]

        rand = random.random()
        role_assignment_mode = "standard"

        if self.reconstruction_ratio > 0 and rand < self.reconstruction_ratio:
            role_assignment_mode = "reconstruction"
            source_wav_path = elem["target_wav_path"]
            if 'target_token_path' in elem:
                semantic_token_path = elem['target_token_path']
        elif self.reversed_ratio > 0 and rand < self.reconstruction_ratio + self.reversed_ratio:
            role_assignment_mode = "reversed"
            source_wav_path = elem["target_wav_path"]
            if 'target_token_path' in elem:
                semantic_token_path = elem['target_token_path']
            target_wav_path = elem["source_wav_path"]
            if 'target_semantic_path' in elem and 'source_semantic_path' in elem:
                semantic_token_path = elem['source_semantic_path']

        try:
            sr = int(cfg["sample_rate"])
            hop = int(cfg["latent_hop_length"])
            seg_dur = float(cfg["segment_duration"])
            hp_cut = float(cfg["highpass_cutoff_freq"])
            align_k = int(cfg["align_multiple"])
            offline = bool(cfg["offline_feature_extracted"])
            ssl_ratio = int(cfg["ssl_per_sem_ratio"])
            feat, ssl_feat, sim_feat = None, None, None

            source_wav = load_audio(source_wav_path, sr, volume_normalize=True, length=None)
            target_wav = load_audio(target_wav_path, sr, volume_normalize=True, length=None)
            if hp_cut != 0:
                source_wav = audio_highpass_filter(source_wav, sr, hp_cut)
                target_wav = audio_highpass_filter(target_wav, sr, hp_cut)
                if source_wav is None or target_wav is None:
                    raise ValueError("highpass returned None")
            
            if offline:
                feat = torch.load(semantic_token_path, weights_only=False)
                ssl_feat = torch.load(ssl_path, weights_only=False)
                T_tok = int(feat.shape[0])
                T_ssl = int(ssl_feat.shape[0])

                if feat is None or ssl_feat is None:
                    raise ValueError("semantic feat / ssl_feat is None")
            else:
                T_tok, T_ssl = None, None

            T_wav = int(len(source_wav) // hop)
            if T_tok is None:
                T = T_wav
            else:
                T_sem_from_ssl = (T_ssl // ssl_ratio) if T_ssl is not None else T_wav
                T = min(T_tok, T_wav, T_sem_from_ssl)
            
            
            length = T // align_k * align_k
            wav_length = length * hop
            source_wav = source_wav[:wav_length]
            target_wav = target_wav[:wav_length]

            if feat is not None:
                feat = feat[:length]

            if ssl_feat is not None:
                ssl_len = length * ssl_ratio
                ssl_feat = ssl_feat[:ssl_len]

            if not self.train:
                cur_dur = (length * hop) / sr
                seg_dur_eff = min(cur_dur, float(cfg["max_val_duration"]))
            else:
                seg_dur_eff = seg_dur

            seg_T = int(sr * seg_dur_eff // hop)
            seg_T = (seg_T // align_k) * align_k
            wav_segment_length = seg_T * hop
            ssl_segment_length = seg_T * ssl_ratio

            if wav_segment_length > wav_length:
                source_wav = np.pad(source_wav, (0, int(wav_segment_length - wav_length)))
                target_wav = np.pad(target_wav, (0, int(wav_segment_length - wav_length)))
                if feat is not None:                    
                    pad_tok = torch.zeros(seg_T - length, dtype=feat.dtype, device=feat.device)
                    feat = torch.cat([feat, pad_tok], dim=0)
                if ssl_feat is not None:
                    Dssl = ssl_feat.shape[1]
                    ssl_pad = torch.zeros(ssl_segment_length - (length * ssl_ratio), Dssl,
                                          dtype=ssl_feat.dtype, device=ssl_feat.device)
                    ssl_feat = torch.cat([ssl_feat, ssl_pad], dim=0)

                start_indice = 0
            
            else:
                if not self.train:
                    start_indice = 0
                else:
                    hi = max(0, length - seg_T)
                    start_indice = random.randint(0, hi)

            source_wav = torch.from_numpy(source_wav)
            target_wav = torch.from_numpy(target_wav)
            max_len = max(source_wav.shape[0], target_wav.shape[0])
            source_wav = F.pad(source_wav, (0, max_len - source_wav.shape[0]))
            target_wav = F.pad(target_wav, (0, max_len - target_wav.shape[0]))
 
            end_indice = start_indice + seg_T
            wav_start_indice = start_indice * hop
            wav_end_indice = end_indice * hop

            feat_segment = feat[start_indice:end_indice] if feat is not None else None
            source_wav_segment = source_wav[wav_start_indice:wav_end_indice]
            target_wav_segment = target_wav[wav_start_indice:wav_end_indice]

            source_wav_cond = source_wav
            target_wav_cond = target_wav
            if self.mask_target_condition:
                mask = torch.ones_like(target_wav_cond)
                mask[wav_start_indice:wav_end_indice] = 0.0
                source_wav_cond = source_wav_cond * mask
                target_wav_cond = target_wav_cond * mask

            if ssl_feat is not None:
                ssl_start_indice = start_indice * ssl_ratio
                ssl_end_indice = end_indice * ssl_ratio
                ssl_feat_segment = ssl_feat[ssl_start_indice:ssl_end_indice]
                ssl_feat_segment = ssl_feat_segment.transpose(0, 1)  # [D, T]
            else:
                ssl_feat_segment = None

            return {
                "index": utt,
                "semantic_tokens": feat_segment.to(torch.long) if feat_segment is not None else None,
                "source_wav": source_wav_segment.float(),
                "target_wav": target_wav_segment.float(),
                "ssl_feat": ssl_feat_segment.float() if ssl_feat_segment is not None else None,
                "sim_feat": sim_feat.float() if sim_feat is not None else None,
                "target_wav_cond": target_wav_cond.float(),
                "source_wav_cond": source_wav_cond.float(),
                "role_assignment_mode": role_assignment_mode,
                # "target_wav_full": target_wav.float(),
            }

        except Exception as e:
            print(f"[VCSSLWAVDataset] Bad case in fetch_data (utt={utt}): {e}")
            return {
                "index": utt,
                "semantic_tokens": None,
                "source_wav": None,
                "target_wav": None,
                "ssl_feat": None,
                "target_wav_cond": None,
                "source_wav_cond": None,
                "role_assignment_mode": None,
            }

    def filter(self, elem: dict):
        """Filter out bad data. Return True if the data is kept."""
        # if elem["semantic_tokens"] is not None:
        if elem["target_wav"] is not None:
            return True
        else:
            return False

    def padding(self, batch: List[dict]):
        """Padding the batch data into training data

        Args:
            batch (List[dict])
        """
        assert isinstance(batch, list)
        collate_batch = {}

        for k in ("index",):
            collate_batch[k] = [b[k] for b in batch]
        collate_batch["role_assignment_mode"] = [b.get("role_assignment_mode") for b in batch]

        for k in ("semantic_tokens", "source_wav", "target_wav", "ssl_feat", "sim_feat"):
            v = [b[k] for b in batch]
            if v[0] is not None:
                pad_value = 0 if k == "semantic_tokens" else 0.0
                collate_batch[k] = stack_tensors_with_aligned_T(v, k, pad_value=pad_value)
                # collate_batch[k] = torch.stack(v, dim=0)
            else:
                collate_batch[k] = None

        if "target_wav_cond" in batch[0]:
            wav_list = [b["target_wav_cond"] for b in batch]
            # pad to longest in batch
            padded_wavs = pad_sequence(wav_list, batch_first=True, padding_value=0.0)
            collate_batch["target_wav_cond"] = padded_wavs.unsqueeze(1)
        else:
            collate_batch["target_wav_cond"] = None
        
        if "source_wav_cond" in batch[0]:
            wav_list = [b["source_wav_cond"] for b in batch]
            # pad to longest in batch
            padded_wavs = pad_sequence(wav_list, batch_first=True, padding_value=0.0)
            collate_batch["source_wav_cond"] = padded_wavs.unsqueeze(1)
        else:
            collate_batch["source_wav_cond"] = None

        return collate_batch
