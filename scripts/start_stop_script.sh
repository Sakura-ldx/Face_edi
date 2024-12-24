#!/bin/bash

# 获取当前小时
current_hour=$(date +%H)

# 如果当前小时大于等于 8 且小于 24，表示在早上 8 点到晚上 12 点之间，启动脚本
if [ $current_hour -ge 8 ] && [ $current_hour -lt 24 ]; then
    echo "Starting script..."
    CUDA_VISIBLE_DEVICES=0 python edit_test_encoder.py --config ../configs/edit/edit_kp_train.yaml
    exit 0
fi

# 如果当前小时大于等于 0 且小于 8，表示在午夜到早上 8 点之间，杀死脚本
if [ $current_hour -ge 0 ] && [ $current_hour -lt 8 ]; then
    echo "Killing script..."
    pkill -f edit_test_encoder.py
    exit 0
fi

