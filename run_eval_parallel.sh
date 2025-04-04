#!/bin/bash

BASE_DIR="/home/vy/datasets/ScanNetVideos"
OUTPUT_DIR="test"
MAX_JOBS=4

job_count=0

for imagedir in "$BASE_DIR"/*/; do
  scene_name=$(basename "$imagedir")
  echo "Launching: $scene_name"

  python main/eval_vlom.py \
    --config-path=../configs \
    --config-name=scannet \
    data.imagedir="$imagedir" \
    data.savedir="$OUTPUT_DIR/$scene_name" &

  ((job_count++))
  if (( job_count % MAX_JOBS == 0 )); then
    wait
  fi
done

wait