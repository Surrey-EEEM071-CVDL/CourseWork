#!/bin/bash

python main.py \
-s cub200 \
-t cub200 \
-a resnet50 \
--height 256 \
--width 256 \
--test-batch-size 100 \
--evaluate \
--save-dir logs/eval-cub200
