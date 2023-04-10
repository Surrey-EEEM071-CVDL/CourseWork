#!/bin/bash

python main.py \
-s veri \
-t veri \
-a mobilenet_v3_small \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 30 \
--stepsize 10 20 \
--train-batch-size 256 \
--test-batch-size 100 \
--save-dir logs/mobilenet_v3_small-veri
