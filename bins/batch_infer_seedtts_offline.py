import argparse
import os
import time

import soundfile as sf
from tqdm import tqdm

from bins.infer_utils import (
    get_seedtts_testset_metainfo,
    load_pair_as_tensors,
    load_xvc,
    run_offline,
    to_numpy_audio,
)


def parse_args():
    parser = argparse.ArgumentParser(description="x-vc batch offline inference on seedtts-testset")
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
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, model, device = load_xvc(args.config, args.ckpt, args.device, args.ema_load)

    metalst = os.path.join("data/seedtts_testset", args.lang, "non_para_reconstruct_meta.lst")
    metainfo = get_seedtts_testset_metainfo(metalst)
    if args.max_items > 0:
        metainfo = metainfo[: args.max_items]

    out_dir = os.path.join(args.save_dir, args.lang, "converted", f"{args.name}_offline")
    os.makedirs(out_dir, exist_ok=True)

    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for utt, _, target_path, _, source_path in tqdm(metainfo):
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

        t0 = time.time()
        recon = run_offline(model, source_wav, target_wav, target_wav_cond)
        infer_sec = time.time() - t0

        recon_np = to_numpy_audio(recon)
        audio_sec = float(recon_np.shape[-1]) / float(cfg["sample_rate"])
        total_infer_sec += infer_sec
        total_audio_sec += audio_sec

        sf.write(os.path.join(out_dir, f"{utt}.wav"), recon_np, samplerate=int(cfg["sample_rate"]))

    total_rtf = total_infer_sec / max(total_audio_sec, 1e-8)
    print(f"saved_dir: {out_dir}")
    print(f"total_rtf: {total_rtf:.6f}")


if __name__ == "__main__":
    main()
