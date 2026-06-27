#!/bin/bash

datasets=(
  eurosat fgvc food101 dtd caltech101
  oxford_flowers oxford_pets stanford_cars sun397 ucf101
)

for dataset in "${datasets[@]}"; do
  for seed in 1 2 3; do
    torchrun --standalone --nproc_per_node=2 main.py \
      --root_path /workspace/ldh/data/FSL \
      --dataset "$dataset" \
      --shots 16 \
      --seed "$seed" \
      --adaptation hydra \
      --setting standard \
      --num_experts 3 \
      --router_temperature 0.1 \
      --router_temperature_schedule fixed \
      --hydra_diversity_weight 0.01 \
      --hydra_balance_weight 0.01 \
      --image_anchor_weight 0.25 \
      --text_anchor_weight 1.0 \
      --save_path outputs
  done
done
