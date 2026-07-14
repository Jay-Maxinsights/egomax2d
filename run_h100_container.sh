#!/bin/bash
# 启动 H100 推理/评估容器（交互式）
cd "$(dirname "$0")"
sudo docker run --gpus all --ipc=host --shm-size=16g \
  --ulimit nofile=65536:65536 \
  --ulimit core=0 \
  -v "$(pwd)":/workspace/egomax2d \
  -v "$(pwd)/data":/workspace/egomax2d/data \
  -v "$(pwd)/work_dirs":/workspace/egomax2d/work_dirs \
  -w /workspace/egomax2d \
  -it egomax2d:h100 bash
