#!/bin/bash
# SGLang Vision (Qwen2.5-VL) launcher — called by GPU Manager start_command.
# Companion to sglang-launch.sh (text LLM on :8001).

CONTAINER_NAME="sglang-vision"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
PORT=8002
MEM_FRACTION=0.50

# Stop any existing container
docker stop "$CONTAINER_NAME" 2>/dev/null
docker rm "$CONTAINER_NAME" 2>/dev/null

# Launch SGLang in detached mode
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus device=1 \
  --ipc=host \
  -p "${PORT}:${PORT}" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:blackwell \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION" \
    --disable-cuda-graph \
    --context-length 8192 \
    --enable-multimodal

echo "[sglang-vision-launch] Container started: $CONTAINER_NAME"
