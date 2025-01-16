#!/bin/bash

# 读取传入的参数

# 根目录
ROOT_PATH="custom_data"
# 待切割数据集文件夹名称
CUSTOM_DATA="dataset"
# 切割碎片保存路径
SAVED_PATH="cut_result"
# 生成的PKL保存路径
PKL_PATH="pkl"

# 原图路径
ARG1="${ROOT_PATH}/${CUSTOM_DATA}"
# 切割碎片路径
ARG2="${ROOT_PATH}/${SAVED_PATH}"
if [ ! -d "$ARG2" ]; then
    mkdir $ARG2
fi


ARG3="${ROOT_PATH}/${PKL_PATH}"
# 1 切割
python3 1_cut_image.py "$ARG1" "$ARG2"

# 2 生成所有碎片数据集
if [ ! -d "$ARG3" ]; then
    mkdir $ARG3
fi
python3 2_get_gt_pair.py "$ARG2" "$ARG3"

# 3 划分 train test val 数据集
python3 3_divide_data.py "$ARG3"

# 4 可视化
python3 4_frag_vis.py "$ROOT_PATH" "$ARG2"
