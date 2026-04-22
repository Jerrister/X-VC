import argparse
import os

import soundfile as sf
from bins.infer_utils import (
    load_pair_as_tensors,
    load_xvc,
    precompute_conditions,
    run_offline,
    run_streaming,
    to_numpy_audio,
)


def parse_args():
    parser = argparse.ArgumentParser(description="x-vc single-pair inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--source_wav_path", type=str, required=True)
    parser.add_argument("--target_wav_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ema_load", action="store_true")
    parser.add_argument("--mask_target_condition", action="store_true")
    parser.add_argument("--latent_hop_length", type=int, default=1280)

    parser.add_argument("--current", type=int, default=0, help="0=offline, >0=streaming")
    parser.add_argument("--chunk", type=int, default=2400)
    parser.add_argument("--future", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, args.ema_load)

    src = os.path.splitext(os.path.basename(args.source_wav_path))[0]
    tgt = os.path.splitext(os.path.basename(args.target_wav_path))[0]

    source_wav, target_wav, target_wav_cond = load_pair_as_tensors(
        source_wav_path=args.source_wav_path,
        target_wav_path=args.target_wav_path,
        cfg=cfg,
        device=device,
        latent_hop_length=args.latent_hop_length,
        mask_target_condition=args.mask_target_condition,
    )

    if args.current == 0:
        recon = run_offline(model, source_wav, target_wav, target_wav_cond)
        mode = "offline"
        save_name = f"{tgt}_{src}_{mode}"
    else:
        speaker_condition, frame_condition = precompute_conditions(model, target_wav, target_wav_cond)
        recon, _ = run_streaming(
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
        mode = "stream"
        save_name = f"{tgt}_{src}_{mode}_chunk_{args.chunk}_current_{args.current}_smooth_{args.smooth}_future_{args.future}"

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{save_name}.wav")
    sf.write(save_path, to_numpy_audio(recon), samplerate=int(cfg["sample_rate"]))

    print(f"source: {args.source_wav_path}")
    print(f"target: {args.target_wav_path}")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
