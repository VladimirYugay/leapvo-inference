#!/bin/bash

BASE_DIR="/home/vy/datasets/ScanNetVideos"
OUTPUT_DIR="test"
SCENE_LIST="scenes.txt"  # file with scene names

# Read each line from the file
while read -r scene_name; do
  imagedir="$BASE_DIR/$scene_name"

  if [ -d "$imagedir" ]; then
    echo "Processing scene: $scene_name"
    echo "$imagedir"

    python main/eval_vlom.py \
      --config-path=../configs \
      --config-name=scannet \
      data.imagedir="$imagedir" \
      data.name=scannet \
      data.savedir="$OUTPUT_DIR/$scene_name"
  else
    echo "Warning: $imagedir does not exist"
  fi
done < "$SCENE_LIST"
