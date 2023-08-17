# 自定义模型

定义新模型时，将模型类型定义为auto，即可加载新模型。其他参数自行更改即可。此处以百川模型为例，加载自定义模型。

如果新加入模型参数配置为AutoModelForCausalLM，AutoTokenizer方式加载，模型类型为auto既可进行测评。
以下是`run_eval.sh`的代码示例：

```text
#baichuan-13b
model_type=auto #模型类型中不存在的，可以使用auto方式进行加载，采用AutoModelForCausalLM,AutoTokenizer方式加载
model_path=/baichuan-13b
exp_name=baichuan13b

exp_date=$(date +"%Y%m%d%H%M%S")
echo "exp_date": $exp_date
output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date
echo "output_path": $output_path

python eval.py \
    --model_type  ${model_type} \
    --model_path ${model_path} \
    ${lora_model:+--lora_model "$lora_model"} \
    --cot True \
    --few_shot True \
    --with_prompt False \
    --ntrain 5 \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --do_save_csv True \
    --do_test False \
    --gpus 0,1,2,3 \
    --only_cpu False \
    --output_dir ${output_path}
```
