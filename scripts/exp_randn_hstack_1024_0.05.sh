#!/bin/bash

# 设置 kdim 的值列表
kdim_values=(1024)
# kdim_values=(1024)
# stds=(0.000001 0.00001 0.0001 0.001 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.35 0.4 0.42, 0.44, 0.46, 0.48, 0.5, 1, 2, 5)
stds=(0.05)

# 循环运行命令
for kdim in "${kdim_values[@]}"; do
    for std in "${stds[@]}"; do
        python commons/train_rff_svc_for_mnist.py cpu --rtype rff --kdim "$kdim" --stdev "$std"
    done
done