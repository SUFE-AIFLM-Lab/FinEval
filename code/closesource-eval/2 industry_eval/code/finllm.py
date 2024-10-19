from abc import ABCMeta, abstractmethod
import re

from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM

import os
import openai 
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
base_url = "https://www.apillm.online/v1"
kwargs = {
    "temperature":0.3
}

class DISCFINLLMBase(metaclass=ABCMeta):

    @abstractmethod
    def generate(self,device: str, prompt: str) -> str:
        # 模型需要接收提示prompt，使用模型生成回复
        raise NotImplementedError

class OpenAILLMGEMINIFLASH(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None, device:str=None, lora_path:str=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        try:
            # 发起 API 调用
            response = self.client.chat.completions.create(model="gemini-1.5-flash", messages=messages, **kwargs)

            # 检查 response 是否有效
            if response is None:
                print("API response is None.")
                return None

            # 检查 response.choices 是否有效
            if not hasattr(response, 'choices') or response.choices is None or len(response.choices) == 0:
                print("API response does not contain 'choices' or 'choices' is empty.")
                return None

            # 提取结果
            result = response.choices[0].message.content

            # 输出结果
            print(result)
            return result

        except Exception as e:
            # 捕获并打印任何异常
            print(f"Error during API call: {e}")
            return None

class OpenAILLMGEMINIPRO(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None, device:str=None, lora_path:str=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        try:
            # 发起 API 调用
            response = self.client.chat.completions.create(model="gemini-1.5-pro", messages=messages, **kwargs)

            # 检查 response 是否有效
            if response is None:
                print("API response is None.")
                return None

            # 检查 response.choices 是否有效
            if not hasattr(response, 'choices') or response.choices is None or len(response.choices) == 0:
                print("API response does not contain 'choices' or 'choices' is empty.")
                return None

            # 提取结果
            result = response.choices[0].message.content

            # 输出结果
            print(result)
            return result

        except Exception as e:
            # 捕获并打印任何异常
            print(f"Error during API call: {e}")
            return None
class OpenAILLMGPT4O(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None, device:str=None, lora_path:str=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        try:
            # 发起 API 调用
            response = self.client.chat.completions.create(model="gpt-4o-mini-2024-07-18", messages=messages, **kwargs)

            # 检查 response 是否有效
            if response is None:
                print("API response is None.")
                return None

            # 检查 response.choices 是否有效
            if not hasattr(response, 'choices') or response.choices is None or len(response.choices) == 0:
                print("API response does not contain 'choices' or 'choices' is empty.")
                return None

            # 提取结果
            result = response.choices[0].message.content

            # 输出结果
            print(result)
            return result

        except Exception as e:
            # 捕获并打印任何异常
            print(f"Error during API call: {e}")
            return None

class OpenAILLMGPT4OMINI(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None, device:str=None, lora_path:str=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        try:
            # 发起 API 调用
            response = self.client.chat.completions.create(model="gpt-4o-mini-2024-07-18", messages=messages, **kwargs)

            # 检查 response 是否有效
            if response is None:
                print("API response is None.")
                return None

            # 检查 response.choices 是否有效
            if not hasattr(response, 'choices') or response.choices is None or len(response.choices) == 0:
                print("API response does not contain 'choices' or 'choices' is empty.")
                return None

            # 提取结果
            result = response.choices[0].message.content

            # 输出结果
            print(result)
            return result

        except Exception as e:
            # 捕获并打印任何异常
            print(f"Error during API call: {e}")
            return None

class OpenAILLMCLAUDESUNNET(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None, device:str=None, lora_path:str=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        try:
            print(messages)
            response = self.client.chat.completions.create(model="claude-3-5-sonnet-20240620", messages=messages, **kwargs)
            print(response)
            result = response.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error during API call: {e}")
            return None

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

#    def generate(self, prompt: str) -> str:
#        answer = self.model.chat(self.tokenizer, prompt)
#        return answer

    def generate(self, prompt:str):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        generation_output = self.model.generate(input_ids=input_ids, max_new_tokens=128)
        return self.tokenizer.decode(generation_output[0])

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

class DISCVFINLLMQwen7BChat(DISCFINLLMBase):
    def __init__(self, model_name_or_path=None,device:str = None, lora_path: str = None):
        model_name_or_path = model_name_or_path
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(device)  # .half().cuda()
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
        template = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: {}\nAssistant: "
        )

        inputs = self.tokenizer([template.format(prompt)], return_tensors="pt")
        inputs = inputs.to(self.device)
        generate_ids = self.model.generate(**inputs, max_new_tokens=256)

        return generate_ids
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
