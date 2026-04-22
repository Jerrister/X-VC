#!/bin/bash

set -euo pipefail

name="xvc_eval"
config="configs/xvc.yaml"
ckpt="ckpts/xvc.pt"
device=0
lang="zh"
save_dir="outputs/seedtts_eval"
max_items=0

python -m bins.batch_infer_seedtts_offline \
  --name "${name}" \
  --config "${config}" \
  --ckpt "${ckpt}" \
  --device "${device}" \
  --lang "${lang}" \
  --save_dir "${save_dir}" \
  --max_items "${max_items}"
