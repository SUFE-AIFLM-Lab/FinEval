# Add a Model(optional)

## Financial Academic Knowledge Evaluation
- The review code is under 1academic_eval
  
- If the model is loaded using AutoModelForCausalLM, AutoTokenizer, specify model_type (model name) as auto, and fill in the rest of the parameters normally to load the new model.

- If the model is loaded in other ways (AutoModelForCausalLM, AutoTokenizer cannot load the model), you can modify the `/code/evaluators/unify_evaluator.py` file

  
1. Customize and add model loading information, modify the `/code/evaluators/unify_evaluator.py` file, and import this parameter at transformers:
   
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
    New way to load models
)
```

2. Add custom model modification information:

```text
MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    "moss":(AutoConfig, AutoTokenizer),
    "Custom model": (model loading method, tokenizer loading method)
}
```
3. Add your new model loading logic in `/code/evaluators/unify_evaluator.py`.


## Financial Industry Knowledge Evaluation
- evaluation code is under folder 2industry_eval

1. Customize and add the class loaded by the model and modify the finllm.py file
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
2. Add custom model modification information and modify the eval.py file
```
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

    'qwen2-72b':Qwen2_7BChat,#new added model
}
```

## Financial Safety Knowledge and Financial Agent Evaluation
- The evaluation code is under 34safety+agenteval. The models are added in the same way as for the financial industry knowledge evaluation.
  
1. add custom model loading classes, modify the finllm.py file
2. Add custom model modification information, modify eval.py file.
