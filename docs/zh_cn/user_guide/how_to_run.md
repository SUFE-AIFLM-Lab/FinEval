# 如何运行

我们继续以Llama-2-7b-hf这个模型为例，进行使用说明的详细解释，我们完成一次测评分四步走。

- 首先在FinEval/code文件夹下放置数据集，并命名为data。

- 下载测评模型权重。

- 修改评测脚本`code/run_eval.sh`的参数。

  运行下面命令，进行修改配置文件

```text
vi run_eval.sh
```

  运行上述命令后，配置文件内容如下。

```text
export PROJ_HOME=$PWD
export KMP_DUPLICATE_LIB_OK=TRUE

# Llama-2-7b-hf模型
# 修改模型名称确定模型权重加载方式，此处默认有五种加载方式，llama,bloom,auto,moss,chatglm,baichuan，一二代模型均支持
model_type=llama 
# 通过huggingface下载的模型权重的位置，此处采用相对位置路径，如果模型路径下载至其他位置，可以使用绝对路径。
model_path=/Llama-2-7b-hf 
# 模型结果生成的目录名称，如果以下参数do_save_csv格式为True,模型运行信息将保存在一个文件夹中，文件夹命名为目录生成名称。
exp_name=Llama-2-7b-hf

exp_date=$(date +"%Y%m%d%H%M%S")
echo "exp_date": $exp_date
output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date
echo "output_path": $output_path

python eval.py \
    --model_type  ${model_type} \
    --model_path ${model_path} \
    ${lora_model:+--lora_model "$lora_model"} \
    --cot False \
    --few_shot True \
    --with_prompt False \
    --ntrain 5 \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --do_save_csv True \
    --do_test False \
    --gpus 0 \ # 测评进行的显卡编号
    --only_cpu False \ # 默认为False，如果为True将使用cpu进行评估，速度会减慢，不推荐使用cpu进行评估。
    --output_dir ${output_path}
```

  

- 运行评测脚本`code/run_eval.sh`

```text
bash run_eval.sh
```

  运行评测脚本后，将会产生每个科目具体的分数以及总的加权分数。
