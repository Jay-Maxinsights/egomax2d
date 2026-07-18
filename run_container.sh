#!/bin/bash
# 启动推理/评估容器（交互式）
# Usage: ./run_container.sh <h100|rtx5090>
#    or: GPU_TYPE=<h100|rtx5090> ./run_container.sh
cd "$(dirname "$0")"

GPU_TYPE="${1:-$GPU_TYPE}"

case "$GPU_TYPE" in
  h100|rtx5090)
    ;;
  *)
    echo "Error: GPU_TYPE must be 'h100' or 'rtx5090' (got: '${GPU_TYPE}')" >&2
    echo "Usage: ./run_container.sh <h100|rtx5090>" >&2
    exit 1
    ;;
esac

sudo docker run --gpus all --ipc=host --shm-size=16g \
  --ulimit nofile=65536:65536 \
  --ulimit core=0 \
  -v "$(pwd)":/workspace/egomax2d \
  -v "$(pwd)/data":/workspace/egomax2d/data \
  -v "$(pwd)/work_dirs":/workspace/egomax2d/work_dirs \
  -w /workspace/egomax2d \
  -it "egomax2d:${GPU_TYPE}" bash
