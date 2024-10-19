import argparse
import torch
import os
import json
import datetime
from evaluate import *
from utils import *
from finllm import *
import nltk
from tqdm import tqdm

nltk.download('punkt')

# 定义模型和评测数据集
model_name = 'claude-3-5-sonnet-20240620'
eval_data = 'all'
gpus = 0
os.environ['PROJ_HOME'] = os.getcwd()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
exp_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
exp_name = model_name
output_path = os.path.join(os.environ['PROJ_HOME'], 'output_dir', exp_name, exp_date)
os.makedirs(output_path, exist_ok=True)

model_lists = {
    'gemini-1.5-flash': OpenAILLMGEMINIFLASH,
    'gemini-1.5-pro': OpenAILLMGEMINIPRO,
    'gpt-4o-2024-08-06': OpenAILLMGPT4O,
    'gpt-4o-mini-2024-07-18': OpenAILLMGPT4OMINI,
    'claude-3-5-sonnet-20240620': OpenAILLMCLAUDESUNNET,
}

Eval_datasets = {
    'finrag': FinRAG, #检索增强
    'fincot': FinCoT, #思维链
    'fintask': FinTASK, #任务分解
    'findiag': FinDiag, #多轮对话
    'findoc': FinDoc, #文档问答
    'apiutil': APIUtil, #API调用
    'apifind': APIFind, #API检索
    #'appsafe': AppSafe, # Application Security
    #'crypsafe': CrypSafe,#
    #'malware': MalWare,
    #'memsafe':MemSafe,
    #'netwrksafe':NetwrkSafe,
    #'pentest':PenTest,
    #'reveng':RevEng,
    #'sftwrsafe':SftwrSafe,
    #'syssafe':SysSafe,
    #'vulnrb':Vulnrb,
    #'websafe':WebSafe,
}

def write_json(filename, data):
    """将数据写入到JSON文件中"""
    output_file = os.path.join(output_path, filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLM种类', default=model_name, choices=model_lists.keys())
    parser.add_argument("--model_name_or_path", type=str, help="模型路径", required=False)
    parser.add_argument('--eval_data', default='all', type=str, help='评测数据集名称')
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', choices=["False", "True"], default="False", help='only use CPU for inference')
    args = parser.parse_args()

    if args.only_cpu == 'True':
        args.gpus = ""
        device = torch.device('cpu')
    else:
        device = torch.device(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model_name_or_path = args.model_name_or_path
    model_name = args.model
    eval_data = args.eval_data

    llm = model_lists[model_name](model_name_or_path, device)

    result_list = []
    if eval_data != 'all':
        assert eval_data in Eval_datasets
        evaluator = Eval_datasets.get(eval_data)

        # 运行评估并生成结果
        result = evaluator().run_evaluation(llm)
        print(result)


        # 生成评分所需的输出文件
        output_file = os.path.join(output_path, f'{eval_data}-output.json')
        if not os.path.exists(output_file):
            # 保存模型输出到指定文件
            write_json(f'{eval_data}-output.json', result)  # 或根据需要修改内容

        # 进行GPT-4评分
        gpt4_scores, ave_score = evaluator().evaluate_gpt4(output_path)
        write_json(f'{model_name}_{eval_data}_gpt4_scores.json', {'scores': gpt4_scores, 'average_score': ave_score})

    else:
        for dataset_name, evaluator in Eval_datasets.items():
            # 运行评估并生成结果
            result = evaluator().run_evaluation(llm)
            result_list.append(result)


            # 生成评分所需的输出文件
            output_file = os.path.join(output_path, f'{dataset_name}-output.json')
            if not os.path.exists(output_file):
                # 保存模型输出到指定文件
                write_json(f'{dataset_name}-output.json', result)  # 或根据需要修改内容

            # 进行GPT-4评分
            gpt4_scores, ave_score = evaluator().evaluate_gpt4(output_path)
            write_json(f'{model_name}_{dataset_name}_gpt4_scores.json', {'scores': gpt4_scores, 'average_score': ave_score})


    print(f"Results saved to {output_path}")
