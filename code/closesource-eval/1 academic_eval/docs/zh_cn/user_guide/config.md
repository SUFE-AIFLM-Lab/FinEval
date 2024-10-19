# 参数配置说明

## 1. 配置说明

few-shot和cot进行参数组合，可以产生四种评测方式:

- few-shot=False且cot=False：即为zero-shot采用只回答答案的方式。
- few-shot=True且cot=False: 即为few-shot采用只回答答案的方式。
- few-shot=False且cot=True: 即为zero-shot方式采用CoT方式回答。
- few-shot=True且cot=True: 即为few-shot方式采用CoT方式回答。

一般来说，pretraining阶段的Base模型few-shot的效果会优于zero-shot，但是经过与人类偏好对齐的Chat模型很可能zero-shot效果会好于Base模型。

不同的model_type代表不同的模型模型读取配置，model_type可从以下配置中选择：
    
```text
"bloom": (BloomForCausalLM, BloomTokenizerFast),
"chatglm": (AutoModel, AutoTokenizer),
"llama": (LlamaForCausalLM, LlamaTokenizer),
"baichuan": (AutoModelForCausalLM, AutoTokenizer),
"auto": (AutoModelForCausalLM, AutoTokenizer),
"moss":(AutoConfig, AutoTokenizer)
```

## 2. 测评参数说明

以下为测评参数说明：
  
```text
('--model_type', default=None, type=str, required=True)：模型名称，此参数必填。
("--model_path", type=str)：模型路径，请填写正确的模型路径
('--lora_model', default="", type=str, help="If None, perform inference on the base model")
("--cot",choices=["False","True"], default="False")：是否采用Chain-of-thought（思维链），如果使用思维链方式将此参数设置为True,默认为False,即不采用思维链方式进行评测。
("--few_shot", choices=["False","True"], default="True")：是否采用few-shot方式进行推理，如果采用此参数将会提供给模型少量例子，供模型学习。
("--ntrain", "-k", type=int, default=5)：few-shot的个数，默认为5-shot，few-shot为False时,此参数失效
("--with_prompt", choices=["False","True"], default="False")：是否采用alpaca的prompt模板，默认不适用，除Alpaca类模型，不建议其他模型设置为True。
("--constrained_decoding", choices=["False","True"], default="True")：是否采用有限制解码方式，由于fineval的评测标准答案为ABCD,提供了两种从模型中提取的答案方案：当constrained_decoding=True,计算模型生成的第一个token分别为ABCD的概率，选择其中概率最大的作为答案；当constrained_decoding=False，用正则表达式从模型生成内容中提取答案。
("--temperature",type=float,default=0.2)：输出温度是一个用来调节生成结果多样性的参数。较低的输出温度值会使得生成的结果更加保守和确定性，更加倾向于选择概率分布中具有较高概率的单词作为输出。换句话说，较低的输出温度会使得生成的结果更加集中和确定。
("--n_times", default=1,type=int)：指定评测的重复次数，将模型放在output_dir下生成指定次数的文件夹，默认为1，生成文件夹为toke0
("--do_save_csv", choices=["False","True"], default="False")：是否将每个科目模型生成结果、标准答案等内容保存在csv文件中。
("--output_dir", type=str)：指定评测结果和csv文件的输出路径
("--do_test", choices=["False","True"], default="False")：在valid和test集上测试,当do_test=False，在valid集上进行测试;当do_test=True,在test集上测试，默认为valid集上进行测试。
('--gpus', default="0", type=str)：模型测试时使用的gpu个数，此处填写gpu编号而非个数。如有多个gpu填写为0,1,2....
('--only_cpu', choices=["False","True"], default="False", help='only use CPU for inference'):是否只采用cpu进行推理，默认不适用cpu进行推理
```
