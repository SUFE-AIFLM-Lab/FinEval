#! /bin/bash
#/*******
##add by weego at 2023-07-24
#1、增加CPU加载模型的选择，参数化配置
#2、增加模型的类型选择，参数化配置
#3、新增支持动态lora权重加载，参数化配置
#4、支持GPU加载的数量设定
#*******/
export KMP_DUPLICATE_LIB_OK=TRUE

#Llama-2-7b-hf模型
# 修改模型名称确定模型权重加载方式，此处默认有五种加载方式，llama,bloom,auto,moss,chatglm,baichuan，一二代模型均支持
model_type=auto
# 通过huggingface下载的模型权重的位置，此处采用相对位置路径，如果模型路径下载至其他位置，可以使用绝对路径。
#lora_model=
# 模型结果生成的目录名称，如果以下参数do_save_csv格式为True,模型运行信息将保存在一个文件夹中，文件夹命名为目录生成名称。
exp_name=statchat

exp_date=$(date +"%Y%m%d%H%M%S")
echo "exp_date": $exp_date
output_path=../results
echo "output_path": $output_path

xuanyuan3_70b=/Llama3-XuanYuan3-70B-Chat

python eval.py --with_prompt False --ntrain 5 --temperature 0.1 --n_times 1 --do_save_csv True --only_cpu False \
    --model_type  ${model_type} \
    --model_path ${xuanyuan3_70b} \
    --cot False \
    --few_shot False \
    --do_test True \
    --gpus 0,1 \
    --output_dir ${output_path}\

python eval.py --with_prompt False --ntrain 5 --temperature 0.1 --n_times 1 --do_save_csv True --only_cpu False \
    --model_type  ${model_type} \
    --model_path ${xuanyuan3_70b} \
    --cot False \
    --few_shot True \
    --do_test True \
    --gpus 0,1 \
    --output_dir ${output_path}\

python eval.py --with_prompt False --ntrain 5 --temperature 0.1 --n_times 1 --do_save_csv True --only_cpu False \
    --model_type  ${model_type} \
    --model_path ${xuanyuan3_70b} \
    --cot True \
    --few_shot False \
    --do_test True \
    --gpus 0,1 \
    --output_dir ${output_path}\

python eval.py --with_prompt False --ntrain 5 --temperature 0.1 --n_times 1 --do_save_csv True --only_cpu False \
    --model_type  ${model_type} \
    --model_path ${xuanyuan3_70b} \
    --cot True \
    --few_shot True \
    --do_test True \
    --gpus 0,1 \
    --output_dir ${output_path}\


