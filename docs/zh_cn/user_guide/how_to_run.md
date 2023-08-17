# 如何运行

我们继续以Llama-2-7b-hf这个模型为例，进行使用说明的详细解释，我们完成一次测评分四步走。

1. 首先在FinEval/code/data文件夹下解压[数据集](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval)。

2. 下载测评模型权重。以下为模型地址：

    百川：
    
    [Baichuan-Chat-13B](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)
    [Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B)
    [Baichuan-Base-13B](https://huggingface.co/baichuan-inc/Baichuan-13B-Base)
    
    LLaMA:
    
    [llama-13b-hf](https://huggingface.co/yahma/llama-13b-hf)
    [Chinese-Alpaca-Plus-7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
    [LLaMA2-chat-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    [LLaMA2-base-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
    [LLaMA2-base-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf)
    [LLaMA2-chat-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
    [LLaMA2-chat-70B]
    [Chinese-llama2](https://huggingface.co/LinkSoul/Chinese-Llama-2-7b)
    [Ziya-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)
    
    ChatGLM:
    
    [ChatGLM](https://huggingface.co/THUDM/chatglm-6b)
    [ChatGLM2](https://huggingface.co/THUDM/chatglm2-6b)
    
    BLOOM:
    
    [Bloomz-7b1-mt](https://huggingface.co/bigscience/bloomz-7b1-mt)
    
    书生 浦语:
    
    InterLm：
    
    [InterLm-chat](https://huggingface.co/internlm/internlm-chat-7b)
    
    猎鹰：
    
    [falcon-7B](https://huggingface.co/tiiuae/falcon-7b)
    [falcon-40B](https://huggingface.co/tiiuae/falcon-40b)
    
    悟道 天鹰
    
    [AquilaChat-7B](https://huggingface.co/BAAI/AquilaChat-7B)
    [Aquila-7B](https://huggingface.co/BAAI/Aquila-7B)
    
    ChatGPT:
    
    [GPT-3.5-turbo]
    [GPT-4]
    
    Moss:
    
    [Moss-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
    [Moss-base](https://huggingface.co/fnlp/moss-moon-003-base)
    
    通义千问：
    
    [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B)
    [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)


3. 修改测评启动脚本`FinEval/code/run_eval.sh`的参数。

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
model_path=Llama-2-7b-hf 
# 模型结果生成的目录名称，如果以下参数do_save_csv格式为True,模型运行信息将保存在一个文件夹中，文件夹命名为目录生成名称。
exp_name=Llama-2-7b-hf
# 如果有训练完的Lora文件，在此加载
exp_name=lora_model

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
    --only_cpu False \ # 默认为False，如果为True将使用cpu进行推理评估，速度会减慢，不推荐使用cpu进行评估。
    --output_dir ${output_path}
```

  

4. 运行评测脚本`code/run_eval.sh`

```text
bash run_eval.sh
```

  运行评测脚本后，将会产生每个科目具体的分数以及总的加权分数。
