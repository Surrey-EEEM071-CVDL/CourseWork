#!/bin/bash

python main.py \
-s cub200 \
-t cub200 \
-a resnet50 \
--height 256 \
--width 256 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet50-cub200
