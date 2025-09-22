#!/bin/bash

# BASE_DIR="/scratch/vyugay/datasets/ARKitVideos/"
# OUTPUT_DIR="arkit_test_set"
# SCENE_LIST="/home/vyugay/projects/vlom/data/split_arkit/test.txt"

# BASE_DIR="/scratch/vyugay/datasets/ScanNetVideos/"
# OUTPUT_DIR="leapvo_inference"
# SCENE_LIST="/home/vyugay/projects/vlom/data/split_scannet/test_vis.txt"

BASE_DIR="/scratch/vyugay/datasets/TUM_RGBD/"
OUTPUT_DIR="tum_eval"
SCENE_LIST="/home/vyugay/projects/vlom/data/split_tum/test.txt"

awk '{print $1}' "$SCENE_LIST" | while read -r scene_name; do
  imagedir="$BASE_DIR/$scene_name"

  if [ -d "$imagedir" ]; then
    echo "Processing: $scene_name"
    python main/eval_vlom.py \
      --config-path=../configs \
      --config-name=scannet \
      data.imagedir="$imagedir" \
      data.name=tum \
      data.savedir="$OUTPUT_DIR/$scene_name"
  else
    echo "Directory not found: $imagedir"
  fi
done
