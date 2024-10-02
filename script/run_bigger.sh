#!/bin/bash
#SBATCH --account=XXXX
#SBATCH --job-name=rwtrain-lm
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --output=output_slimpajam_large_%A.txt

set -x
which python
python --version
python -c 'import torch;print("Pytorch version: ",torch.__version__);print("Cuda version: ",torch.version.cuda)'
nvidia-smi

# setup HF cache directories
export HF_HOME=/LLMs_cache/misc
export HF_DATASETS_CACHE=/LLMs_cache/datasets
export TRANSFORMERS_CACHE=/LLMs_cache/models

cd ..

# gpt2 large
#torchrun --nproc_per_node=auto src/run.py --config_json config/methods/ours_nodr_684m.json --wandb_proj loss-reweighting --wandb_run GPT2_LARGE_ours_extremes_b48 --total_iterations 20000
#torchrun --nproc_per_node=auto src/run.py --config_json config/methods/baseline_684m.json --wandb_proj loss-reweighting --wandb_run GPT2_LARGE_baseline_b48 --total_iterations 20000


# Domain Rewighting + Ours with gpt2 large
#torchrun --nproc_per_node=auto src/run.py --config_json config/methods/ours_doge_large.json --wandb_proj loss-reweighting-extras --wandb_run GPT2_LARGE_ours_doge --total_iterations 20000
#torchrun --nproc_per_node=auto src/run.py --config_json config/methods/ours_doremi_large.json --wandb_proj loss-reweighting-extras --wandb_run GPT2_LARGE_ours_doremi --total_iterations 20000
#torchrun --nproc_per_node=auto src/run.py --config_json config/methods/doge.json --wandb_proj loss-reweighting-extras --wandb_run GPT2_LARGE_doge --total_iterations 20000
#torchrun --nproc_per_node=auto src/run.py --config_json config/methods/doremi.json --wandb_proj loss-reweighting-extras --wandb_run GPT2_LARGE_doremi --total_iterations 20000

