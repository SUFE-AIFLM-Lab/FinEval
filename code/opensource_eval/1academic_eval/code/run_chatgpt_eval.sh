#! /bin/bash
export PROJ_HOME=$PWD
export KMP_DUPLICATE_LIB_OK=TRUE

# input your openai api key
api_key=
base_url=
exp_name=gpt-3.5-turbo-1106
exp_date=$(date +"%Y%m%d%H%M%S")
output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date

echo "exp_date": $exp_date
echo "output_path": $output_path

python eval_chatgpt.py \
    --api_key ${api_key} \
    --base_url ${base_url} \
    --cot False \
    --few_shot False \
    --n_times 1 \
    --ntrain 5 \
    --do_test True \
    --do_save_csv True \
    --output_dir ${output_path} \
    --model_name gpt-3.5-turbo-1106
