#!/bin/bash

python main.py \
-s veri \
-t veri \
-a mobilenet_v3_small \
--height 224 \
--width 224 \
--test-batch-size 100 \
--evaluate \
--save-dir logs/eval-mobilenet_v3_small-veri
