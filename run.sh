#!/bin/bash

python3 mondrian_k_anonymity_implementation.py \
  --data_path adult_dataset/adult.data \
  --output_dir results \
  --k_values 2 5 10 20 50 100 \
  --cnn_epochs 10 \
  --cnn_batch_size 128 \
  --cnn_learning_rate 0.001
