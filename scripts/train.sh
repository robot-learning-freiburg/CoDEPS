#!/bin/bash

python_env="codeps"

conda activate $python_env;
alias python="BIN_PATH/python";
export WANDB_API_KEY="WANDB_API_KEY";

#pip install wandb --upgrade
wandb login --relogin WANDB_API_KEY;

CUDA_VISIBLE_DEVICES="0, 1" \
OMP_NUM_THREADS=4 \
torchrun --nproc_per_node=2 --master_addr='IP' \
                            --master_port=22001`` train_codeps.py \
                            --mode=train \
                            --run_name='RUN_NAME' \
                            --project_root_dir="ROOT_DIR" \
                            --filename_defaults_config=default_config.py \
                            --comment="Train CoDEPS." \
                            --filename_config=train_cityscapes.yaml \
#                            --resume=""

conda deactivate;
