#!/bin/bash

# 设置环境变量
export PROJ_HOME=$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE

# 获取当前日期时间
exp_date=$(date '+%Y%m%d%H%M%S')

# 设置OpenAI API key和base_url
api_key=$OPENAI_API_KEY
base_url='https://www.apillm.online/v1'

# 如果未传入模型名称，使用默认值
exp_name=${1:-'claude-3-5-sonnet-20240620'}

# 设置输出路径
output_path="$PROJ_HOME/output_dir/$exp_name/$exp_date"

# 创建输出目录
mkdir -p "$output_path"

# 打印输出信息
echo "exp_date: $exp_date"
echo "output_path: $output_path"
echo "model_name: $exp_name"

# 构建并执行命令
python eval_chatgpt.py \
  --api_key "$api_key" \
  --base_url "$base_url" \
  --cot "True" \
  --few_shot "True" \
  --n_times "1" \
  --ntrain "5" \
  --do_test "True" \
  --do_save_csv "True" \
  --output_dir "$output_path" \
  --model_name "$exp_name"
