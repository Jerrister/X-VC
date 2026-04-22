import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from models.codec.sac.utils import inference_factory, process_audio
from utils.file import load_config


def _to_device(device_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def load_xvc(config_path: str, ckpt_path: str, device_id: int, ema_load: bool):
    cfg = load_config(config_path)
    if "config" in cfg:
        cfg = cfg["config"]

    args = {
        "config": config_path,
        "ckpt": ckpt_path,
        "device": _to_device(device_id),
        "ema_load": ema_load,
    }
    model = inference_factory(cfg, args)
    return cfg, model, args["device"]


def load_pair_as_tensors(
    source_wav_path: str,
    target_wav_path: str,
    cfg: dict,
    device: torch.device,
    latent_hop_length: int,
    mask_target_condition: bool,
):
    source_wav_np = process_audio(source_wav_path, cfg, latent_hop_length)
    target_wav_np = process_audio(target_wav_path, cfg, latent_hop_length)

    source_wav = torch.from_numpy(source_wav_np).unsqueeze(0).unsqueeze(1).float().to(device)
    target_wav = torch.from_numpy(target_wav_np).unsqueeze(0).unsqueeze(1).float().to(device)

    if mask_target_condition:
        sr = int(cfg["sample_rate"])
        mask_cond_pad = torch.zeros((1, 1, int(2.4 * sr)), device=device)
        target_wav_cond = torch.cat([target_wav, mask_cond_pad], dim=-1)
    else:
        target_wav_cond = target_wav

    return source_wav, target_wav, target_wav_cond


def _required(mod, name: str):
    if not mod:
        raise RuntimeError(f"x-vc infer requires `{name}`.")
    return mod


@torch.inference_mode()
def precompute_conditions(model, target_wav: torch.Tensor, target_wav_cond: torch.Tensor):
    speaker_encoder = _required(model.speaker_encoder, "speaker_encoder")
    mel_extractor = _required(model.mel_extractor, "mel_extractor")
    speaker_condition, _ = speaker_encoder(target_wav)
    frame_condition = mel_extractor(target_wav_cond)
    return speaker_condition, frame_condition


@torch.inference_mode()
def run_offline(model, source_wav: torch.Tensor, target_wav: torch.Tensor, target_wav_cond: torch.Tensor):
    outputs = model.inference(
        {
            "source_wav": source_wav,
            "target_wav": target_wav,
            "target_wav_cond": target_wav_cond,
        }
    )
    return outputs["recons"]


@torch.inference_mode()
def run_stream_chunk_forward(
    model,
    source_wav: torch.Tensor,
    speaker_condition: torch.Tensor,
    frame_condition: torch.Tensor,
):
    semantic_encoder = _required(model.semantic_encoder, "semantic_encoder")
    semantic_adapter = _required(model.semantic_adapter, "semantic_adapter")
    acoustic_encoder = _required(model.acoustic_encoder, "acoustic_encoder")
    acoustic_quantizer = _required(model.acoustic_quantizer, "acoustic_quantizer")
    prenet = _required(model.prenet, "prenet")
    acoustic_converter = _required(model.acoustic_converter, "acoustic_converter")
    acoustic_decoder = _required(model.acoustic_decoder, "acoustic_decoder")

    feat = semantic_encoder.extract_and_encode(source_wav.squeeze(1))["speech_tokens"]
    sem_emb = semantic_encoder.embed_ids(feat)
    sem_emb = semantic_adapter(sem_emb.transpose(1, 2)).transpose(1, 2)

    z = acoustic_encoder(source_wav)
    aq_outputs = acoustic_quantizer(z)
    zq = aq_outputs[0]
    acu_emb = zq.transpose(1, 2)

    combined_emb = torch.cat([sem_emb, acu_emb], dim=2)
    x = prenet(combined_emb.transpose(1, 2), speaker_condition)
    x = acoustic_converter(x, frame_condition, speaker_condition)
    y = acoustic_decoder(x)
    return y


@torch.inference_mode()
def run_streaming(
    model,
    source_wav: torch.Tensor,
    speaker_condition: torch.Tensor,
    frame_condition: torch.Tensor,
    sample_rate: int,
    chunk_ms: int,
    current_ms: int,
    future_ms: int,
    smooth_ms: int,
):
    if current_ms <= 0:
        raise ValueError("`current_ms` must be > 0 for streaming mode.")

    source_len = source_wav.shape[-1]
    history_ms = chunk_ms - current_ms - smooth_ms - future_ms
    if history_ms < 0:
        raise ValueError("Invalid streaming window: chunk_ms - current_ms - smooth_ms - future_ms must be >= 0.")

    overlap_len = smooth_ms * sample_rate // 1000
    if overlap_len > 0:
        device = source_wav.device
        fade_in = 0.5 * (1 - torch.cos(torch.pi * torch.linspace(0, 1, overlap_len, device=device)))
        fade_out = 1 - fade_in
        tail_buffer = torch.zeros(1, 1, overlap_len, device=device)
    else:
        fade_in = fade_out = tail_buffer = None

    current_len = current_ms * sample_rate // 1000
    total_n_chunks = (source_len + current_len - 1) // current_len
    recon_wav_list: List[torch.Tensor] = []
    latency_ms_list: List[float] = []

    for i in range(total_n_chunks):
        t0 = time.time()

        start_len = (i * current_ms - history_ms) * sample_rate // 1000
        end_len = (i * current_ms + current_ms + smooth_ms + future_ms) * sample_rate // 1000
        left_pad = max(0, -start_len)
        right_pad = max(0, end_len - source_len)

        chunk_wav = source_wav[:, :, start_len + left_pad : end_len - right_pad]
        chunk_wav = F.pad(chunk_wav, (left_pad, right_pad), mode="constant", value=0)

        chunk_out = run_stream_chunk_forward(model, chunk_wav, speaker_condition, frame_condition)
        cur_start = history_ms * sample_rate // 1000
        cur_end = (history_ms + current_ms) * sample_rate // 1000
        chunk_recon_wav = chunk_out[:, :, cur_start:cur_end]

        if overlap_len > 0:
            if i > 0:
                head = chunk_recon_wav[..., :overlap_len]
                head_smoothed = tail_buffer * fade_out + head * fade_in
                chunk_recon_wav = torch.cat([head_smoothed, chunk_recon_wav[..., overlap_len:]], dim=-1)
            tail_start = (history_ms + current_ms) * sample_rate // 1000
            tail_buffer = chunk_out[:, :, tail_start : tail_start + overlap_len]

        recon_wav_list.append(chunk_recon_wav)
        latency_ms_list.append((time.time() - t0) * 1000.0)

    recon_wav = torch.cat(recon_wav_list, dim=-1)
    recon_wav = recon_wav[:, :, :source_len]
    return recon_wav, latency_ms_list


def get_seedtts_testset_metainfo(metalst: str) -> List[Tuple[str, str, str, str, str]]:
    with open(metalst, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metainfo = []
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = parts
        elif len(parts) == 4:
            utt, prompt_text, prompt_wav, gt_text = parts
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", f"{utt}.wav")
        else:
            continue

        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        if not os.path.isabs(gt_wav):
            gt_wav = os.path.join(os.path.dirname(metalst), gt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


def to_numpy_audio(wav_tensor: torch.Tensor) -> np.ndarray:
    return wav_tensor.squeeze().detach().cpu().numpy()
