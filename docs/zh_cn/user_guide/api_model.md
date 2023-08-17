# 基于 API 的模型

以OpenAi的api模型为例，使用`code/run_chatgpt_eval.sh`脚本文件运行评测。除openai_key参数之外，其他参数与基于模型权重的测评参数一致，核心参数为model_name和openai_key。

model_name名称处请填写正确的模型名称，若测评gpt4，请规范填写，最终评测脚本中请填写为gpt-4。正确的模型信息请参考OpenAI官网。

此外此处openai_key建议使用付费版本，5美元版本有速度限制影响测评。

```text
#! /bin/bash
export PROJ_HOME=$PWD
export KMP_DUPLICATE_LIB_OK=TRUE

# 确定api的key
openai_key=sk-***************** #此处填写您的openai_key进行测评

exp_name=chatgpt
exp_date=$(date +"%Y%m%d%H%M%S")
output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date

echo "exp_date": $exp_date
echo "output_path": $output_path

python eval_chatgpt.py \
    --openai_key ${openai_key} \
    --cot False \
    --few_shot False \
    --n_times 1 \
    --ntrain 5 \
    --do_test False \
    --do_save_csv False \
    --output_dir ${output_path} \
    --model_name gpt-4 # 请填写正确的OpenAI模型名称
```

使用 [FastChat](https://github.com/lm-sys/FastChat) 提供开源 LLM 模型的 API，可支持OpenAI API接口形式模型测评。

