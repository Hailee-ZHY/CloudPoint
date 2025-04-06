#!/bin/bash

echo "Training process"
python train.py --epoches 20 --batch_size 8 --lr 0.001 

echo "Test Results"
python main.py  