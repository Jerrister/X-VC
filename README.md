# X-VC

[![arXiv](https://img.shields.io/badge/arXiv-2604.12456-b31b1b.svg)](https://arxiv.org/abs/2604.12456)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-fcd022?logo=huggingface&logoColor=000)](https://huggingface.co/chenxie95/X-VC)
[![Demo Page](https://img.shields.io/badge/Demo-Project%20Page-blue)](https://x-vc.github.io)

Official code release for **X-VC: Zero-shot Streaming Voice Conversion in Codec Space**.

![X-VC overview](figures/overview.png)

## Environment Setup

### 1. Clone

```bash
git clone https://github.com/Jerrister/X-VC.git
cd X-VC
```

### 2. Create conda environment (Python 3.10) and install dependencies

```bash
conda create -n xvc python=3.10 -y
conda activate xvc
pip install -U pip
pip install -r requirements.txt
```

### 3. Prepare pretrained assets (for inference and training)

Prepare:
- [GLM-4-Voice-Tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer) (for semantic tokenization)
- [ERes2Net speaker encoder](https://modelscope.cn/models/iic/speech_eres2net_sv_en_voxceleb_16k) (for speaker feature extraction)

Then set paths in [`configs/xvc.yaml`](configs/xvc.yaml), especially:
- `model.generator.semantic_encoder.encoder.from_pretrained`
- `model.generator.semantic_encoder.cfg`
- `model.generator.speaker_encoder.pretrained_dir`

### 4. Prepare checkpoints

Put checkpoints under `ckpts/`, for example:

```text
ckpts/
  xvc.pt
```

## Inference

### Single-pair Inference

Use [`scripts/infer_single.sh`](scripts/infer_single.sh).

```bash
bash scripts/infer_single.sh
```

Key arguments in this script:
- `current=0` for offline inference.
- `current>0` for streaming inference.
- `chunk/current/future/smooth` control streaming behavior.

Outputs are saved under `save_dir` (default: `outputs/xvc_single`).

### Batch Offline Inference (SeedTTS testset example)

Use [`scripts/batch_infer_seedtts_offline.sh`](scripts/batch_infer_seedtts_offline.sh).

```bash
bash scripts/batch_infer_seedtts_offline.sh
```

This script reports:
- `saved_dir`
- `total_rtf`

### Batch Streaming Inference (SeedTTS-eval as example)

Use [`scripts/batch_infer_seedtts_stream.sh`](scripts/batch_infer_seedtts_stream.sh).

```bash
bash scripts/batch_infer_seedtts_stream.sh
```

This script reports:
- `saved_dir`
- `avg_latency_ms`

## Training

### Step 1: Prepare pretrained dependencies

Before training, prepare the required pretrained dependencies:
- [SAC pretrained checkpoint(s)](https://huggingface.co/Soul-AILab/SAC-16k-62_5Hz) (for model initialization)

Then set corresponding paths in [`configs/xvc.yaml`](configs/xvc.yaml), especially:
- `model.generator.checkpoint`
- `model.discriminator.checkpoint`

### Step 2: Prepare training data

Organize your training/validation data in JSONL format and set:
- `datasets.train`
- `datasets.val`

in [`configs/xvc.yaml`](configs/xvc.yaml).

### Step 3: Modify training configs

You can adjust training behavior in:
- [`configs/xvc.yaml`](configs/xvc.yaml) (main training config)
- [`configs/ds_stage2.json`](configs/ds_stage2.json) (DeepSpeed config)

### Step 4: Start training

Use [`scripts/train.sh`](scripts/train.sh).

```bash
bash scripts/train.sh
```

Notes:
- Default training engine is DeepSpeed (`configs/ds_stage2.json`).
- Main experiment config is `configs/xvc.yaml`.
- Set your `WANDB_API_KEY` in `scripts/train.sh` before running if you use wandb logging.

## Data Format

Training config points to JSONL files in `configs/xvc.yaml`:
- `datasets.train`
- `datasets.val`

Each JSONL line should be a JSON object.

Required fields:
- `target_utt`
- `source_wav_path`
- `target_wav_path`

Optional field:
- `source_utt`

Minimal example:

```json
{"source_utt":"utt_0001","source_wav_path":"<path_to_source>","target_utt":"utt_0002","target_wav_path":"<path_to_target>"}
```

## Acknowledgements

This codebase builds upon open-source components from [SAC](https://github.com/Soul-AILab/SAC) and the broader audio generation ecosystem.

## Citation

If you find our work useful, please cite our work as:

```bibtex
@misc{zheng2026xvczeroshotstreamingvoice,
      title={X-VC: Zero-shot Streaming Voice Conversion in Codec Space}, 
      author={Qixi Zheng and Yuxiang Zhao and Tianrui Wang and Wenxi Chen and Kele Xu and Yikang Li and Qinyuan Chen and Xipeng Qiu and Kai Yu and Xie Chen},
      year={2026},
      eprint={2604.12456},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2604.12456},
}
```
