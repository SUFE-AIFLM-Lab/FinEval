import os
from tqdm import tqdm
import openai
from evaluators.evaluator import Evaluator
from time import sleep
import re

from openai import OpenAI

class ChatGPT_Evaluator(Evaluator):
    def __init__(self, choices, k, api_key,model_name,base_url):
        super(ChatGPT_Evaluator, self).__init__(choices, model_name, k)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def format_example(self,line,include_answer=True,cot=False):
        example=line['question']
        for choice in self.choices:
            example+=f'\n{choice}. {line[f"{choice}"]}'

        example+='\n答案：'
        if include_answer:
            if cot:
                ans=line["answer"]
                content="让我们一步一步思考，你可以适当展示你的思路\n"+f"\n所以答案是{ans}。"
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":content}
                ]
            else:
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":line["answer"]}
                ]
        else:
            if cot:
                example += "\n让我们一步步思考,\n"
                return [
                        {"role":"user","content":example}
                        ]
            else:
                return [
                    {"role":"user","content":example},
                ]
    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt=[
            {
                "role":"system",
                "content":f"你是一个中文人工智能助手，以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案，请不要输出*，你只需要生成答案，不需要解释其他错误选项和总结。"
            }
        ]
        k=self.k
        if self.k==-1:
            k=dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True,cot=cot)
            if i==0:
                tmp[0]["content"]=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案，请不要输出*，你只需要生成答案，不需要解释其他错误选项和总结。\n\n"+tmp[0]["content"]
            prompt+=tmp
        return prompt

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None,cot=False):
        correct_num = 0
        all_answer = {}
        if save_result_dir:
            result = []
            score=[]
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df,cot=cot)
        else:
            few_shot_prompt=[
                {
                    "role":"system",
                    "content":f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案，请不要输出*，你只需要生成答案，不需要解释其他错误选项和总结。"
                }
            ]    
        answers = list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
            question = self.format_example(row, include_answer=False,)
            full_prompt = few_shot_prompt + question
            if not few_shot:
                full_prompt[-1]["content"]=f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案，请不要输出*，你只需要生成答案，不需要解释其他错误选项和总结。\n\n"+full_prompt[-1]["content"]
            response=None
            timeout_counter=0
            while response is None and timeout_counter<=30:
                try:
                    sleep(1.5)
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=full_prompt,
                        temperature=0.3
                    )
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter+=1
                    print(msg)
                    sleep(5)
                    continue
            if response==None:
                response_str=""
            else:
                response_str = response.choices[0].message.content

               # response_str = response['choices'][0]['message']['content']
            if cot:
              if not few_shot:
                response_str=response_str.strip()
                if few_shot:
                    if len(response_str)>0:
                        if self.exact_match(response_str,row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
                else:
                    if len(response_str)>0:
                        ans_list=self.extract_ans(response_str)
                        if len(ans_list)>0 and (ans_list[-1]==row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
              else:
                ans_list=re.findall(r"答案是(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"答案为(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"选项(.+?)是正确的。",response_str)

                if len(ans_list)==0:
                    correct=0
                else:
                    if self.exact_match(ans_list[-1],row["answer"]):
                        correct_num+=1
                        correct=1
                    else:
                        correct=0

            else:
                response_str=response_str.strip()
                if few_shot:
                    if len(response_str)>0:
                        if self.exact_match(response_str,row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
                else:
                    if len(response_str)>0:
                        ans_list=self.extract_ans(response_str)
                        if len(ans_list)>0 and (ans_list[-1]==row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
            if save_result_dir:
                result.append(response_str)
                score.append(correct)
            all_answer[str(row_index)] = row["answer"]
        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output']=result
            test_df["correctness"]=score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'),encoding="utf-8",index=False)
        return correct_ratio,all_answer



    def extract_ans(self, response_str):
        patterns = [
            r"答案\s?:?\s?([A-D])",  # Matches "答案: A"
            r"答案为\s?:?\s?([A-D])",  # Matches "答案为: A"
            r"答案是\s?:?\s?([A-D])",  # Matches "答案是: A"
            r"选项\s?([A-D])",  # Matches "选项 A"
            r"答案选\s?([A-D])",  # Matches "答案选 A"
            r"答案应为\s?:?\s?([A-D])",  # Matches "答案应为: A"
            r"正确的一项是\s?([A-D])",  # Matches "正确的一项是 A"
            r"答案\s?[：:\s]?([A-D])",  # Matches "答案：A" or "答案:A"
            r"选项\s?([A-D])\s?是正确的",  # Matches "选项 A 是正确的"
            r"答案为\s?[：:\s]?([A-D])",  # Matches "答案为：A"
            r"答案选\s?[：:\s]?([A-D])",  # Matches "答案选：A"
            r"答案是\s?[：:\s]?([A-D])",  # Matches "答案是：A"
            r"答案应该是\s?[：:\s]?([A-D])",  # Matches "答案应该是：A"
            r"答案[:：]\s*\*\*([A-D])\.?\s"
        ]

        ans_list = []

        # Check if response_str starts with a single character option
        if response_str and response_str[0] in ["A", "B", "C", "D"]:
            ans_list.append(response_str[0])

        # Try each pattern to extract the answer
        for pattern in patterns:
            if not ans_list:
                matches = re.findall(pattern, response_str)
                if matches:
                    ans_list.extend(matches)

        # Remove any duplicates
        ans_list = list(set(ans_list))

        # Debugging output
        print(f"Response String: {response_str}")
        print(f"Extracted Answers: {ans_list}")

        return ans_list

