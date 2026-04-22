#!/bin/bash

# export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY=""  # Set your wandb key before running
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0

script_dir=$(dirname "$(realpath "$0")")

# Get the root directory
root_dir="$(realpath "$script_dir/..")"

# Set the run name
run_name="xvc_16khz"

# Set default parameters
log_dir="exp/$run_name"
nnodes=1
nproc_per_node=8
# nproc_per_node=1
num_workers=16
config="configs/xvc.yaml"
train_engine="deepspeed"
deepspeed_config="configs/ds_stage2.json"
resume_step=0
debug=true
project='x-vc'

port=10101

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --nnodes)
      nnodes="$2"
      shift 2
      ;;
    --gpus|--nproc_per_node)
      nproc_per_node="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done


cd "$root_dir" || exit
source utils/parse_options.sh

# Check if log_dir is already an absolute path
if [[ "$log_dir" != /* ]]; then
    log_dir="$root_dir/$log_dir"
fi

# Check if log directory exists
if [ $resume_step -eq 0 ]; then
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
        echo "Log directory created: $log_dir"
    elif [ "$debug" = false ]; then
        echo "Error: Log directory '$log_dir' already exists. Please remove or choose a different directory."
        exit 1
    fi
fi

# Write command to run.bash
tag="$(date +'%Y%m%d_%H%M%S')"

cat <<EOT > "$log_dir/${tag}_run.sh"
#!/bin/bash

# # Change directory to the root directory
cd "$root_dir" || exit

torchrun --nnodes=${nnodes} --nproc_per_node=${nproc_per_node} --master_port=${port} \\
        -m bins.train \\
        --config ${config} \\
        --log_dir ${log_dir} \\
        --train_engine ${train_engine} \\
        --deepspeed_config ${deepspeed_config} \\
        --resume_step ${resume_step} \\
        --date ${tag} \\
        --project ${project} \\
        --enable_wandb \\
        --wandb_runs_name ${run_name} \\
        # --checkpoint ${checkpoint} \\
EOT

chmod +x "$log_dir/${tag}_run.sh"
echo "run bash is saved to $log_dir/${tag}_run.sh"

echo "execute $log_dir/${tag}_run.sh"

bash "$log_dir/${tag}_run.sh"

# Example command:
# bash scripts/train.sh
# bash scripts/train.sh --nnodes 1 --gpus 4
