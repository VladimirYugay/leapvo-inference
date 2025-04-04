#!/bin/bash

BASE_DIR="/home/vy/datasets/ScanNetVideos"
OUTPUT_DIR="test"

for imagedir in "$BASE_DIR"/*/; do
  scene_name=$(basename "$imagedir")

  echo "Processing scene: $scene_name"
  echo "$imagedir"

  python main/eval_vlom.py \
    --config-path=../configs \
    --config-name=scannet \
    data.imagedir="$imagedir" \
    data.name=scannet \
    data.savedir="$OUTPUT_DIR/$scene_name"
done
