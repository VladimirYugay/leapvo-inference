#!/bin/bash

BASE_DIR="/scratch/vyugay/datasets/ARKitVideos/"
OUTPUT_DIR="test"
SCENE_LIST="/home/vyugay/projects/vlom/data/split_arkit/test.txt"

awk '{print $1}' "$SCENE_LIST" | while read -r scene_name; do
  imagedir="$BASE_DIR/$scene_name"

  if [ -d "$imagedir" ]; then
    echo "Processing: $scene_name"
    python main/eval_vlom.py \
      --config-path=../configs \
      --config-name=scannet \
      data.imagedir="$imagedir" \
      data.name=arkit \
      data.savedir="$OUTPUT_DIR/$scene_name"
  else
    echo "Directory not found: $imagedir"
  fi
done
