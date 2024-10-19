#!/bin/bash

# 设置环境变量
export PROJ_HOME=$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE

# 获取当前日期时间
exp_date=$(date '+%Y%m%d%H%M%S')

# 如果未传入模型名称，使用默认值
model_name=${1:-'claude-3-5-sonnet-20240620'}

# 如果未传入评估数据集名称，使用默认值
eval_data=${2:-'all'}



# 设置输出路径
output_path="$PROJ_HOME/output_dir/$model_name/$exp_date"

# 创建输出目录
mkdir -p "$output_path"

# 打印调试信息
echo "Model Name: $model_name"
echo "Evaluation Dataset: $eval_data"
echo "Model Path: $model_path"
echo "Output Path: $output_path"
echo "Execution Date: $exp_date"

# 运行Python评估脚本
python eval.py \
    --model "$model_name" \
    --eval_data "$eval_data" \
    --gpus "0" \
    --only_cpu "False"

# 检查Python脚本的执行情况
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: $output_path"
else
    echo "Evaluation failed. Please check the logs for details."
fi

