#!/bin/bash

set -euo pipefail

config="configs/xvc.yaml"
ckpt="ckpts/xvc.pt"
device=0

source_wav_path="examples/source.wav"
target_wav_path="examples/target.wav"
save_dir="outputs/xvc_single"

# 0 = offline; >0 = streaming
current=00
chunk=2400
future=100
smooth=20

python -m bins.infer_single \
  --config "${config}" \
  --ckpt "${ckpt}" \
  --device "${device}" \
  --source_wav_path "${source_wav_path}" \
  --target_wav_path "${target_wav_path}" \
  --save_dir "${save_dir}" \
  --current "${current}" \
  --chunk "${chunk}" \
  --future "${future}" \
  --smooth "${smooth}"
