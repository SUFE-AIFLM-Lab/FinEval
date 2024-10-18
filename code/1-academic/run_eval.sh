#!/bin/bash
# 2023-07-24 by weego
# 使用OpenAI API运行模型

export PROJ_HOME=~/MyData/data/guoxin/evaluation/fineval-1/FinEval
export KMP_DUPLICATE_LIB_OK=TRUE

# 设置OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# 设置实验名称和输出路径
exp_name=statchat
exp_date=$(date +"%Y%m%d%H%M%S")
echo "exp_date": $exp_date
output_path=~/MyData/data/guoxin/evaluation/fineval-1/FinEval/results/${exp_name}/$exp_date
echo "output_path": $output_path

# 运行Python脚本，调用OpenAI API
python eval.py \
    --model_type "openai_api" \
    --cot False \
    --few_shot False \
    --with_prompt True \
    --ntrain 5 \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --do_save_csv True \
    --do_test False \
    --gpus 0 \
    --only_cpu True \
    --output_dir ${output_path}
