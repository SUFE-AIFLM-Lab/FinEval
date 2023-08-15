# 参数配置说明

## 1. 配置说明

few-shot和cot进行参数组合，可以产生四种评测方式:

- few-shot为False cot为False：即为zero-shot采用只回答答案的方式。
- few-shot为True  cot为False: 即为few-shot采用只回答答案的方式。
- few-shot为False cot为True: 即为zero-shot方式采用CoT方式回答。
- few-shot为True cot为True: 即为few-shot方式采用CoT方式回答。

一般来说，pretraining阶段的模型few-shot 的效果总是会比 zero-shot 好一些，但是经过instruction tuning之后的模型，且instruction tuning没有few-shot data的话，很可能zero-shot会更好。

不同的model_type代表不同的模型模型读取配置，model_type请从以下模型中选择：
    
```text
"bloom": (BloomForCausalLM, BloomTokenizerFast),
"chatglm": (AutoModel, AutoTokenizer),
"llama": (LlamaForCausalLM, LlamaTokenizer),
"baichuan": (AutoModelForCausalLM, AutoTokenizer),
"auto": (AutoModelForCausalLM, AutoTokenizer),
"moss":(AutoConfig, AutoTokenizer)
```

## 2. 模型配置信息

以下为模型配置信息：
  
```text
--model_type 模型名称
--model_path 模型路径
--cot  是否采用Chain-of-thought
--few_shot  是否采用few-shot学习
--with_prompt  是否采用alpaca的prompt模板，默认不适用
--ntrain few-shot的个数，few-shot为False,此参数失效
--constrained_decoding 是否采用有限制解码方式，由于fineval的评测标准答案为ABCD,提供了两种从模型中提取的答案方案：当constrained_decoding=True,计算模型生成的第一个token分别为ABCD的概率，选择其中概率最大的作为答案；当constrained_decoding=False，用正则表达式从模型生成内容中提取答案。
--temperature 模型解码的温度
--n_times 指定评测的重复次数，将模型放在output_dir下生成指定次数的文件夹，默认为1，生成文件夹为toke0
--do_save_csv 是否将模型生成结果、提取的答案等内容保存在csv文件中
--do_test 在valid和test集上测试,当do_test=False，在valid集上进行测试;当do_test=True,在test集上测试
--gpus 模型测试时使用的gpu个数
--only_cpu True 是否只采用cpu进行评估
--output_dir 指定评测结果的输出路径

```
