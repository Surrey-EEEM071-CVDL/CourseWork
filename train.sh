#!/bin/bash

python main.py \
-s veri \
-t veri \
-a resnet18 \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet18-veri
