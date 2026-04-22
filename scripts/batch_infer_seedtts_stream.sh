#!/bin/bash

set -euo pipefail

name="xvc_eval"
config="configs/xvc.yaml"
ckpt="ckpts/xvc.pt"
device=0
lang="zh"
save_dir="outputs/seedtts_eval"
max_items=0

chunk=2400
current=120
future=100
smooth=20

python -m bins.batch_infer_seedtts_stream \
  --name "${name}" \
  --config "${config}" \
  --ckpt "${ckpt}" \
  --device "${device}" \
  --lang "${lang}" \
  --save_dir "${save_dir}" \
  --max_items "${max_items}" \
  --chunk "${chunk}" \
  --current "${current}" \
  --future "${future}" \
  --smooth "${smooth}"
