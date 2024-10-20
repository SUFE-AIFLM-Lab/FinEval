# 支持新模型（选用）
我们将介绍开源评测代码中如何支持新模型的评测，代码在code/opensource_eval中

## 金融学术知识评测
- 评测代码在1academic_eval下
  
- 如果模型采用AutoModelForCausalLM,AutoTokenizer方式加载,指定model_type（模型名称）为auto，其余参数正常填写，即可加载新模型。

- 如果模型采用其他方式加载(AutoModelForCausalLM,AutoTokenizer无法加载模型)，可修改`/code/evaluators/unify_evaluator.py`文件


1. 自定义增加模型加载信息,修改`/code/evaluators/unify_evaluator.py`文件，在transformers处进行导入此参数：

```text
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoConfig,
    模型新的加载方式
)
```

2. 加入自定义模型修改信息		

```text
MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    "moss":(AutoConfig, AutoTokenizer),
    "自定义模型":(模型加载方式,分词器加载方式)
}
```

3. 在`/code/evaluators/unify_evaluator.py`中加入您新的模型加载逻辑。

## 金融行业知识评测
- 评测代码在2industry_eval下
  
1. 自定义增加模型加载的类,修改`finllm.py`文件：
```
   class Qwen2_7BChat(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype,
                                               device_map="auto")
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()
    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        return response
```
2. 加入自定义模型修改信息，修改`eval.py`文件
   model_lists = {
    "disc":DISCVFINLLMBaichuan13BBase,
    "chatglm4-9b":GLM49B,
    'internlm2.5-20b-chat':DISCVFINLLMInternLm2Chat20B,
    'baichuan2-13b-chat': DISCVFINLLMBaichuan13BChat,
    'cfgpt2-7b':CFGPT2_7B,
    'yi-9b':YiChat,
    'yi-34b':YiChat,
    'xuanyuan2-70b':XuanYuan2_70B,
    'xuanyuan3-70b':XuanYuan2_70B,

    'qwen2-72b':Qwen2_7BChat,#新增模型
}

## 金融安全知识和金融智能体评测
- 评测代码在34safety+agenteval下
    模型的添加方式和金融行业知识评测一样。
  1. 自定义增加模型加载的类,修改`finllm.py`文件
     
  2. 加入自定义模型修改信息，修改`eval.py`文件
  

