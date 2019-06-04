#!/usr/bin/env bash

# 1. 20190602 V100
python main.py -j 8 --epochs 300 --lr 0.2 --schedule 150 225 --train-batch 256 --test-batch 100

# 2. 20190602 4*V100
#python main.py -j 20 --epochs 300 --lr 0.3 --schedule 150 225 --train-batch 512 --test-batch 200