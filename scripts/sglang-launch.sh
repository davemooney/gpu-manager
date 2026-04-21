#!/bin/bash
# SGLang LLM Inference Server — Docker launcher
# Called by GPU Manager start_command. Runs Docker in detached mode.

MODEL_ENV="/home/aidin/.config/vllm/model.env"
VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
if [ -f "$MODEL_ENV" ]; then
    source "$MODEL_ENV"
fi

CONTAINER_NAME="sglang-llm"
PORT=8001
MEM_FRACTION=0.30
EXTRA_ARGS=""

# Vision-language models need --enable-multimodal
case "$VLLM_MODEL" in
    *VL*|*vl*|*vision*)
        EXTRA_ARGS="--enable-multimodal"
        ;;
esac

echo "[sglang-launch] Model: $VLLM_MODEL"
echo "[sglang-launch] Extra args: $EXTRA_ARGS"

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
    --model-path "$VLLM_MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --mem-fraction-static "$MEM_FRACTION" \
    --disable-cuda-graph \
    --context-length 4096 \
    $EXTRA_ARGS

echo "[sglang-launch] Container started: $CONTAINER_NAME"
