#!/bin/bash
# Multi-GPU Distributed Training Launch Script using Accelerate

accelerate launch --config_file accelerate_config.yaml train_llama.py "$@"
