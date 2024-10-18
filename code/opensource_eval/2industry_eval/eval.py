import argparse
from evaluate import *
from utils import *
from finllm import *

model_lists = {
    #'chatglm-6b': DISCVFINLLMChatGLM6B,
    #'chatglm2-6b': DISCVFINLLMChatGLM26B,
    'baichuan-7b': DISCVFINLLMBaichuan7B,
    'baichuan-13b-base': DISCVFINLLMBaichuan13BBase,
    'baichuan-13b-chat': DISCVFINLLMBaichuan13BChat,
    'bloomz-7b': DISCVFINLLMBloomz7B,
    "chatglm3-6b":GLM49B,
    'internlm2-20b-chat':DISCVFINLLMInternLm2Chat20B,
   # 'llama2-70b-chat':DISCVFINLLMLLAMA2CHAT70B,
   # 'fingpt':FinGPTv3,
    'gpt4':OpenAILLM,

    #最新一轮FinEval
    "disc":DISCVFINLLMBaichuan13BBase,
    "chatglm4-9b":GLM49B,
    'internlm2.5-20b-chat':DISCVFINLLMInternLm2Chat20B,
    'baichuan2-13b-chat': DISCVFINLLMBaichuan13BChat,
    'cfgpt2-7b':CFGPT2_7B,
    'yi-9b':YiChat,
    'yi-34b':YiChat,
    'qwen2-7b':Qwen2_7BChat,
    'qwen2-72b':Qwen2_7BChat,
    'xuanyuan2-70b':XuanYuan2_70B,
    'xuanyuan3-70b':XuanYuan2_70B,
    'gpt-4o-2024-08-06':OpenAILLM,
    'gpt-4o-mini-2024-07-18':OpenAILLM,
    'claude-3-5-sonnet-20240620':OpenAILLM,
    'gemini-1.5-flash':OpenAILLM,
    'gemini-1.5-pro':OpenAILLM,
}

Eval_datasets = {
    #objective
    'finnl': FinNLEvaluator,  # 金融文本分类，非生成类任务，多标签分类 
    "finfetext": FinFETextEvaluator,  # 金融情感分析，非生成式任务，多分类 #FSA
    'finre': FinREEvaluator,  # 关联关系分类，非生成类任务，多标签分类 #RE
    'finqa': FinQAEvaluator,  # 金融事件抽取，问答，生成类任务 #FEE
    'finnsp1': FinNSP1Evaluator,  # 负面主体判定，非生成类任务，多标签分类
    'fincqa': FinCQAEvaluator,  # 因果事件抽取，生成类任务
    #subjective
    'finna': FinNAEvaluator,  # 金融摘要，生成类任务
    "fincustomer":FinCustomerEvaluator,# 金融用户画像，多分类
    "finsales":FinSalesEvaluator, # 金融营销术语解释，生成性任务
    "finsuggestion":FinSuggestionEvaluator,# 金融建议，生成任务

    #"finterm":FinTermEvaluator,# 金融术语，生成式任务
    # "finqc": FinQCEvaluator,  # 金融问题分类，非生成式认识，多分类
    # 'finfeforum': FinFEForumEvaluator,  # 论坛情感分析，非生成类任务，多分类
    # 'finnsp2': FinNSP2Evaluator,  # 负面主体判定，非生成类任务，多标签分类
    
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLM种类')
    parser.add_argument("--model_name_or_path", type=str, help="模型路径",required=False)
    parser.add_argument('--lora_path', default='', type=str, help='LORA路径')
    parser.add_argument('--eval_data', default='all', type=str, help='评测数据集名称')
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', choices=["False", "True"], default="False", help='only use CPU for inference')
    parser.add_argument('--cot', action='store_true', help='是否使用思维链解题')

    args = parser.parse_args()
    print(args)
    if args.only_cpu == 'True':
        args.gpus = ""
        device = torch.device('cpu')
    else:
        device = torch.device(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(device)
    model_name_or_path = args.model_name_or_path
    model_name = args.model
    lora_path = None if args.lora_path == '' else args.lora_path
    eval_data = args.eval_data

    # 加载模型
    llm = model_lists[model_name](model_name_or_path, device, lora_path)

    output_dir = f'output/{model_name}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if eval_data != 'all':
        assert eval_data in Eval_datasets
        evaluator = Eval_datasets.get(eval_data)
        output_zero, output_few, result = evaluator(args.cot).run_evaluation(llm)
        datasetname = evaluator().dataset
        suffix = '-cot' if args.cot else ''
        suffix2 = '_lora' if lora_path else ''
        
        write_json(f'{output_dir}/{datasetname}_{suffix2}zeroshot{suffix}.json', output_zero)
        write_json(f'{output_dir}/{datasetname}_{suffix2}fewshot{suffix}.json', output_few)
        print('{}结果：\n{}'.format(datasetname,result))
    else:
        result_list = []
        for _, evaluator in Eval_datasets.items():
            output_zero, output_few, result = evaluator(args.cot).run_evaluation(llm)
            result_list.append(result)
            datasetname = evaluator().dataset
            suffix = '-cot' if args.cot else ''
            suffix2 = '_lora' if lora_path else ''
            
            # 保存zeroshot&fewshot输出结果
            write_json(f'{output_dir}/{datasetname}{suffix2}_zeroshot{suffix}.json', output_zero)
            write_json(f'{output_dir}/{datasetname}{suffix2}_fewshot{suffix}.json', output_few)
            # 保存评测分数
            write_json(f'{model_name}{suffix2}{suffix}_eval.json', result_list)




