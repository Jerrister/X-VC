import argparse
import os
import numpy as np

import soundfile as sf
from tqdm import tqdm

from bins.infer_utils import (
    get_seedtts_testset_metainfo,
    load_pair_as_tensors,
    load_xvc,
    precompute_conditions,
    run_streaming,
    to_numpy_audio,
)


def parse_args():
    parser = argparse.ArgumentParser(description="x-vc batch streaming inference on seedtts-testset")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--ema_load", action="store_true")
    parser.add_argument("--mask_target_condition", action="store_true")
    parser.add_argument("--latent_hop_length", type=int, default=1280)
    parser.add_argument("--max_items", type=int, default=0, help="0 means all")
    parser.add_argument("--swap", action="store_true")

    parser.add_argument("--chunk", type=int, default=2400)
    parser.add_argument("--current", type=int, required=True, help="must be > 0")
    parser.add_argument("--future", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.current <= 0:
        raise ValueError("Streaming script requires `--current > 0`.")

    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, args.ema_load)

    metalst = os.path.join("data/seedtts_testset", args.lang, "non_para_reconstruct_meta.lst")
    metainfo = get_seedtts_testset_metainfo(metalst)
    if args.max_items > 0:
        metainfo = metainfo[: args.max_items]

    run_name = (
        f"{args.name}_stream"
        f"_chunk_{args.chunk}"
        f"_current_{args.current}"
        f"_smooth_{args.smooth}"
        f"_future_{args.future}"
    )
    out_dir = os.path.join(args.save_dir, args.lang, "converted", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # requirement: skip the first sample in final latency stats
    per_utt_latency = []

    for idx, (utt, _, target_path, _, source_path) in enumerate(tqdm(metainfo)):
        if args.swap:
            source_path, target_path = target_path, source_path

        source_wav, target_wav, target_wav_cond = load_pair_as_tensors(
            source_wav_path=source_path,
            target_wav_path=target_path,
            cfg=cfg,
            device=device,
            latent_hop_length=args.latent_hop_length,
            mask_target_condition=args.mask_target_condition,
        )
        speaker_condition, frame_condition = precompute_conditions(model, target_wav, target_wav_cond)

        recon, latency_list = run_streaming(
            model=model,
            source_wav=source_wav,
            speaker_condition=speaker_condition,
            frame_condition=frame_condition,
            sample_rate=int(cfg["sample_rate"]),
            chunk_ms=args.chunk,
            current_ms=args.current,
            future_ms=args.future,
            smooth_ms=args.smooth,
        )

        sf.write(
            os.path.join(out_dir, f"{utt}.wav"),
            to_numpy_audio(recon),
            samplerate=int(cfg["sample_rate"]),
        )

        if idx > 0 and len(latency_list) > 0:
            per_utt_latency.append(float(np.mean(latency_list)))

    avg_latency = float(np.mean(per_utt_latency)) if len(per_utt_latency) > 0 else 0.0
    print(f"saved_dir: {out_dir}")
    print(f"avg_latency_ms: {avg_latency:.6f}")


if __name__ == "__main__":
    main()
