import os
import re
from tqdm import tqdm
import random
import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoConfig,
    AutoModelForSeq2SeqLM
)
from transformers.generation import GenerationConfig
from accelerate import init_empty_weights,load_checkpoint_and_dispatch
from evaluators.evaluator import Evaluator
MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    "moss":(AutoConfig, AutoTokenizer)
}
from vllm import LLM, SamplingParams


class unify_Evaluator(Evaluator):
    def __init__(self, choices, k, device, model_type, model_path, lora_model='', temperature=0.2, cot=False):
        super(unify_Evaluator, self).__init__(choices, model_path, k)
        load_type = torch.float16
        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        device_map = 'auto' if self.device != torch.device('cpu') else None

        if cot == True:
            self.model = LLM(
                model=self.model_path, 
                trust_remote_code=True,
                #enable_lora=True,
                tensor_parallel_size=2,
            )
            return
        if model_type != "moss":            
            model_class, tokenizer_class = MODEL_CLASSES[model_type]
            self.tokenizer = tokenizer_class.from_pretrained(model_path, trust_remote_code=True)
            self.base_model = model_class.from_pretrained(
                model_path,
                load_in_8bit=False,
                torch_dtype=load_type,
                low_cpu_mem_usage=True,
                device_map=device_map,
                trust_remote_code=True,
                )
        else:
            self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, torch_dtype=torch.float16, trust_remote_code=True)
            self.model.tie_weights()
            self.model = load_checkpoint_and_dispatch(self.model, model_path, device_map="auto",no_split_module_classes=["MossBlock"],dtype=torch.float16)
        
        if lora_model:
            if model_type != "moss":
                self.model = PeftModel.from_pretrained(self.base_model, lora_model, torch_dtype=load_type, device_map='auto')
                print(f"loaded lora model:"+lora_model)
            else:
                self.model = PeftModel.from_pretrained(self.model,lora_model,torch_dtype=load_type,device_map="auto")
                print(f"loaded lora model:"+lora_model)
        else:
            if model_type != "moss":
                self.model = self.base_model
            else:
                pass
        if device == torch.device('cpu'):
            self.model.float()
        
        self.generation_config = dict(
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=512
        )

        self.sA_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.sB_id = self.tokenizer.encode("B", add_special_tokens=False)[0]
        self.sC_id = self.tokenizer.encode("C", add_special_tokens=False)[0]
        self.sD_id = self.tokenizer.encode("D", add_special_tokens=False)[0]
        self.A_id = self.tokenizer.encode("：A")[-1]
        self.B_id = self.tokenizer.encode("：B")[-1]
        self.C_id = self.tokenizer.encode("：C")[-1]
        self.D_id = self.tokenizer.encode("：D")[-1]

    def eval_subject_vllm(self, subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        cot=False,
        save_result_dir=None,
        with_prompt=False,
        constrained_decoding=False,
        do_test=False
        ):
        #stop_tokens = ["USER:", "USER", "ASSISTANT:", "ASSISTANT", "### Instruction:", "### Instruction", "Response:", "Response"]
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=512)

        all_answers = {}
        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        if few_shot:
            history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            history = ''
        answers = list(test_df['answer'])
        #answers = ['NA'] * len(test_df) if do_test is True else list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot,with_prompt=with_prompt)
            instruction = history + '你只需要回答下面的一道题目，回答后立即停止输出。\n' + question
            print('+' * 80)
            print(instruction) #打印最终的propmt
            print('+'*80)
            if with_prompt:
                prompt_template = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Response: ")

                instruction = prompt_template.format_map({'instruction': instruction,'subject':subject_name})

            generation_output = self.model.generate(instruction,sampling_params)
            response = generation_output[0].outputs[0].text
            print('模型回答是\n{}'.format(response))
            ans, direct_extract = self.extract_answer(row, response)
            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0
            print(f"\n=======begin {str(row_index)}=======")
            print("question: ", question)
            print("response: ", response)
            print("ans: ", ans)
            print("ground truth: ", answers[row_index], "\n")
            if save_result_dir:
                result.append(response)# cot时response进行extract_answer方法提取出ans
                score.append(correct)
            print(f"=======end {str(row_index)}=======")

            all_answers[str(row_index)] = ans

        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output'] = result
            test_df['correctness'] = score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'))

        return correct_ratio, all_answers

    def eval_subject(self, subject_name,
            test_df,
            dev_df=None,
            few_shot=False,
            cot=False,
            save_result_dir=None,
            with_prompt=False,
            constrained_decoding=False,
            do_test=False):
        all_answers = {}
        if constrained_decoding is True:
            self.generation_config['output_scores'] = True
            self.generation_config['return_dict_in_generate'] = True
            self.generation_config['max_new_tokens'] = 1
            self.generation_config['top_p'] = 1.0
            self.generation_config['top_k'] = 0

        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        if few_shot:
            history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            history = ''
        answers = list(test_df['answer'])
        #answers = ['NA'] * len(test_df) if do_test is True else list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot,with_prompt=with_prompt)
            instruction = history + question
            # print('+' * 80)
            # print(instruction) #打印最终的propmt
            # print('+'*80)
            if with_prompt:
                prompt_template = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Response: ")

                instruction = prompt_template.format_map({'instruction': instruction,'subject':subject_name})


            inputs = self.tokenizer(instruction, return_tensors="pt")
            model_type = self.model_type
            if model_type != "chatglm":
                try:
                    generation_output = self.model.generate(
                        input_ids = inputs["input_ids"].to(self.device),
                        attention_mask = inputs['attention_mask'].to(self.device),
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **self.generation_config
                    )
                except Exception as e:
                    self.model.generation_config = GenerationConfig.from_pretrained(self.model_path, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
                    generation_output = self.model.generate(
                    input_ids = inputs["input_ids"].to(self.device),
                    attention_mask = inputs['attention_mask'].to(self.device),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self.generation_config
                    )
            else:
                generation_output = self.model.generate(
                    input_ids = inputs["input_ids"].to(self.device),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self.generation_config
                )
            
            batch_size, length = inputs.input_ids.shape
            if constrained_decoding is True:
                logits = generation_output.scores[0][0]
                logits = logits.float().cpu().detach()
                choices1_logits = logits[[self.sA_id,self.sB_id,self.sC_id,self.sD_id]]
                choices2_logits = logits[[self.A_id,self.B_id,self.C_id,self.D_id]]
                choicesAll_logits = (choices1_logits + choices2_logits).numpy()
                assert not (np.any(np.isinf(choicesAll_logits)) or np.any(np.isnan(choicesAll_logits)))
                ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choicesAll_logits)]
                response = self.tokenizer.decode([logits.argmax(-1).item()])
            else:
                response = self.tokenizer.decode(generation_output[0, length:], skip_special_tokens=True)
                ans, direct_extract = self.extract_answer(row, response)
            if ans == answers[row_index]:
                correct_num += 1
                correct = 1
            else:
                correct = 0
            print(f"\n=======begin {str(row_index)}=======")
            print("question: ", question)
            print("response: ", response)
            print("ans: ", ans)
            print("ground truth: ", answers[row_index], "\n")
            if save_result_dir:
                result.append(response)# cot时response进行extract_answer方法提取出ans
                score.append(correct)
            print(f"=======end {str(row_index)}=======")

            all_answers[str(row_index)] = ans

        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output'] = result
            test_df['correctness'] = score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'))

        return correct_ratio, all_answers

    def format_example(self, line, include_answer=True, cot=False, with_prompt=False):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                #example += "让我们一步一步思考，\n" + \
                example += "\n请一步一步思考，写出解题过程，最后输出答案。\n" + \
                line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
            else:
                example += '\n答案：' + line["answer"] + '\n\n'
        else:
            if with_prompt is False:
                if cot:
                    #example += "\n让我们一步一步思考，\n1."
                    example += "\n请一步一步思考，写出解题过程，最后输出答案。\n" 
                else:
                    example += '\n答案：'
            else:
                if cot:
                    example += "\n答案是什么？让我们一步一步思考，\n1."
                else:
                    example += '\n答案是什么？ '
        return example

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是中国关于{subject}考试的单项选择题，我会给你几个有答案的例子，请你根据最后一个题目的要求进行回答。\n你只需要回答最后一个题目\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(
                dev_df.iloc[i, :],
                include_answer=True,
                cot=cot
            )
        return prompt

    def extract_answer(self, line, gen_ans):
        m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
        if len(m) > 0 and m[-1] in self.choices:
            return m[-1], True
        answer_patterns = [
            r'([ABCD])是正确的',
            r'选项([ABCD])正确',
            r'答案为([ABCD])',
            r'答案是([ABCD])',
            r'答案([ABCD])',
            r'选择([ABCD])',
            r'答案：([ABCD])',
            r'选择答案([ABCD])',
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
            r"选项(.+?)是正确的。",
            r"答案为(.+?)。"
        ]
        # RE extraction
        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer, False
        # only containing one choice-character
        m = re.findall(r'[ABCD]', gen_ans, re.M)
        if len(m) >= 1:
            answer = m[0]
            print('only containing one choice-character')
            return answer, False
        # only containing one choice-context
        choices_dict = {}
        pattern = ""
        for c in self.choices:
            choices_dict[str(line[f'{c}'])] = c
            pattern += re.escape(str(line[f'{c}']))+"|"
        pattern = pattern[:-1]
        m = re.findall(pattern, gen_ans, re.M)
        print("w/ escape:",repr(pattern),gen_ans,(len(m)>=1))
        if len(m) >= 1:
            answer = choices_dict[m[0]]
            return answer, False
        return  random.choice('ABCD'), False
