#!/bin/bash
# Lambda Lab A100
MODEL_PATH="./vllm_model"
PORT=5000

vllm-server --model $MODEL_PATH \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.85 \
            --max-num-batched-tokens 5120 \
            --quantization awq \
            --port $PORT