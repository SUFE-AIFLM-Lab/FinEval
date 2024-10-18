from abc import ABCMeta, abstractmethod
import re

from transformers import AutoTokenizer, AutoModel
import transformers
from peft import PeftModel, PeftConfig
import torch
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.generation.utils import GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
from vllm import LLM ,SamplingParams
import os
import openai 
from openai import OpenAI


class DISCFINLLMBase(metaclass=ABCMeta):

    @abstractmethod
    def generate(self,device: str, prompt: str) -> str:
        # 模型需要接收提示prompt，使用模型生成回复
        raise NotImplementedError


class YiChat(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True,use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(device)  # .half().cuda()

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()
    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(
            input_ids.to('cuda'),
            max_new_tokens=512,
        )
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(response)
        return response

class FinMA_7B(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None, device: str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                                         trust_remote_code=True,
                                                         torch_dtype=dtype).to(device)
        if lora_path:
            peft_model_id = lora_path
            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()

    def generate(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        generation_output = self.model.generate(
            input_ids=input_ids, 
            max_new_tokens=512,
            #pad_token_id=self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(generation_output[0])
        #response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #response = self.tokenizer.decode(generation_output[0][input_ids.shape[1]:],skip_special_tokens=True)
        print(response)
        return response

class CFGPT2_7B(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).cuda()  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()
    def generate(self, prompt: str) -> str:
        respone, history = self.model.chat(
            self.tokenizer, prompt, history=[]
        )
        print(respone)
        return respone

class Qwen2_7BChat(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=0.1, top_p=0.95,max_tokens=512)
        self.model = LLM(model=model_name_or_path, dtype="float16", quantization="gptq")

    def generate(self, prompt: str) -> str:
        # Prepare your prompts
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # generate outputs
        outputs = self.model.generate([text], self.sampling_params)

        for output in outputs:
            #prompt = output.prompt
            generated_text = output.outputs[0].text
        print(generated_text)
        return generated_text

class XuanYuan2_70B(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):

        self.sampling_params = SamplingParams(temperature=0.1, top_p=0.95,max_tokens=1024)
        self.model = LLM(model=model_name_or_path, dtype="float16", tensor_parallel_size=2)

    def generate(self, prompt: str):
        system_message = "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与 与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
        seps = [" ", "</s>"]
        roles = ["Human", "Assistant"]

        prompt = system_message + seps[0] + roles[0] + ": " + prompt + seps[0] + roles[1] + ":"
        #print(f"输入: {content}")
        result = self.model.generate(prompt, self.sampling_params)
        result_output = [[output.outputs[0].text, output.outputs[0].token_ids] for output in result]
        response = result_output[0][0]
        print(f"{response}")
        return response

class DISCVFINLLMXUANYUAN70BCHAT(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=False, legacy = True, trust_remote_code=True)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype, device_map="auto")  # .half().cuda()
       # self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

       # if lora_path:
       #     peft_model_id = lora_path
    
       #     self.model = PeftModel.from_pretrained(self.model, peft_model_id)

        self.model.eval()

    def generate(self, prompt:str):
        system_message = "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与 与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
        seps = [" ","</s>"]
        roles = ["Human", "Assitant"]

        content = ""
        prompt = system_message + seps[0] + roles[0] + ": " + content + seps[0] + roles[1] + ":"
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        generation_output = self.model.generate(**inputs,  max_new_tokens=128, do_sample = True, temperature = 0.7, top_p=0.95)
        return self.tokenizer.decode(generation_output.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

class OpenAILLM(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str=None, lora_path:str=None):
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

    def generate(self, prompt, **kwargs):
        messages = [{"role":"user", "content":prompt}]
        print(messages)
        response = self.client.chat.completions.create(model = "gpt-4-1106-preview",messages=messages, **kwargs)
        print(response)
        result = response.choices[0].message.content
        return result

class DISCVFINLLMLLAMACHAT70B(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype, device_map="auto")  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)

    def generate(self, prompt:str):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        generation_output = self.model.generate(input_ids=input_ids, max_new_tokens=128)
        return self.tokenizer.decode(generation_output[0])


class DISCVFINLLMDISCFINLLM(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(device)  # .half().cuda()
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
       
        if lora_path:
            peft_model_id = lora_path
            self.model = PeftModel.from_pretrained(self.model, peft_model_id)

        self.model = self.model.eval()
    def generate(self, prompt: str) -> str:
        answer, history = self.model.chat(self.tokenizer, prompt, history=None)
        return answer

class DISCVFINLLMInternLm2Chat20B(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).cuda()  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()
    def generate(self, prompt: str) -> str:
        respone, history = self.model.chat(self.tokenizer, prompt, history=[])
        return respone

class DISCVFINLLMChatGLM26B(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(device)  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()

    def generate(self, prompt: str) -> str:
        answer, history = self.model.chat(self.tokenizer, prompt, history=[])
        return answer

class DISCVFINLLMChatGLM6B(DISCFINLLMBase):
    def __init__(self,model_name_or_path=None ,device: str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(device)  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()

    def generate(self, prompt: str) -> str:
        answer, history = self.model.chat(self.tokenizer, prompt, history=[])
        return answer


class GLM49B(DISCFINLLMBase):
    def __init__(self,model_name_or_path=None ,device: str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(device)  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

        inputs = inputs.to('cuda')
        gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer


class DISCVFINLLMBaichuan13BBase(DISCFINLLMBase):
    def __init__(self,model_name_or_path = None, device: str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True).to(device)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.device = device

    def generate(self, prompt: str) -> str:
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, messages)
        print(response)
        return response

class DISCVFINLLMChatGLM36B(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(device)  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()

    def generate(self, prompt: str) -> str:
        answer, history = self.model.chat(self.tokenizer, prompt, history=[])
        return answer

class DISCVFINLLMBaichuan13BChat(DISCFINLLMBase):
    def __init__(self, model_name_or_path = None,device: str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True).to(device)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
            print('lora加载完！')

    def generate(self, prompt: str) -> str:
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, messages)

        return response

class DISCVFINLLMBaichuan7B(DISCFINLLMBase):
    def __init__(self, model_name_or_path = None,device: str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).half()
        self.model = self.model.to(device)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)

        self.device = device

    def generate(self, prompt: str) -> str:
        template = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: {}\nAssistant: "
        )

        inputs = self.tokenizer(template.format(prompt), return_tensors='pt')
        inputs = inputs.to(self.device)
        pred = self.model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        answer = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        print(answer)
        pattern = answer.split('Assistant: ', 1)

        assistant_text = pattern[-1]
        print(assistant_text)
        return assistant_text

class DISCVFINLLMBloomz7B(DISCFINLLMBase):
    def __init__(self, model_name_or_path = None,device: str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).half().to(device)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.device = device

    def generate(self, prompt: str) -> str:

        template = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: {}\nAssistant: "
        )
        inputs = self.tokenizer.encode_plus(template.format(prompt), return_tensors='pt')
        outputs = self.model.generate(**inputs.to(self.device), max_new_tokens=128, repetition_penalty=1.1)
        answer = self.tokenizer.decode(outputs[0])
        pattern = r'Assistant: (.+?)(?:</s>|$)'
        matches = re.findall(pattern, answer)
        # 输出结果
        if matches != []:
            assistant_text = matches[0]
        else:
            assistant_text = '无'

        return assistant_text

class FinGPTv3:
    def __init__(self, device: str = None):
        model_name_or_path = "/data/sufeModel/Model/chatglm2-6b"
        peft_model = "/data/sufeModel/Model/FinGPT_ChatGLM2"
        dtype = torch.float16
        # 训练后的lora保存的路径
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
        self.model = PeftModel.from_pretrained(self.model, peft_model)
        self.device = device


    def generate(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt, return_tensors='pt', padding=True, max_length=512)
        res = self.model.generate(**tokens.to(self.device), max_length=512)
        res_sentences = self.tokenizer.decode(res[0])
        answer = res_sentences.replace(prompt, '').strip()

        return answer


if __name__ == '__main__':
    pass
