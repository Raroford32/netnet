#!/bin/bash

# Set the number of GPUs to use
NUM_GPUS=8

# Run the distributed training script
torchrun --nproc_per_node=$NUM_GPUS main.py --action train
