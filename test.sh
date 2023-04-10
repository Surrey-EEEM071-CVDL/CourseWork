#!/bin/bash

python main.py \
-s veri \
-t veri \
-a resnet18 \
--height 224 \
--width 224 \
--test-batch-size 100 \
--evaluate \
--save-dir logs/eval-resnet18-veri
