import os
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
    "chatglm3-6b":DISCVFINLLMChatGLM36B,
    'internlm2-20b-chat':DISCVFINLLMInternLm2Chat20B,
   # 'llama2-70b-chat':DISCVFINLLMLLAMA2CHAT70B,
   # 'fingpt':FinGPTv3,
    'gpt4':OpenAILLM,

    #最新一轮FinEval
    "disc":DISCVFINLLMBaichuan13BBase,
    "chatglm4-9b":GLM49B,
    'internlm2.5-20b-chat':DISCVFINLLMInternLm2Chat20B,
    'baichuan2-13b-chat': DISCVFINLLMBaichuan13BChat,
    'finma-7b':FinMA_7B,
    'cfgpt2-7b':CFGPT2_7B,
    'yi-9b':YiChat,
    'yi-34b':YiChat,
    'qwen2-7b':Qwen2_7BChat,
    'qwen2-72b':Qwen2_7BChat,
    'xuanyuan2-70b-chat':XuanYuan2_70B,
    'xuanyuan3-70b-chat':XuanYuan2_70B,
}

Eval_datasets = {
    'finrag': FinRAG, #检索增强
    'fincot': FinCoT, #思维链
    'fintask': FinTASK, #任务分解
    'findiag': FinDiag, #多轮对话
    'findoc': FinDoc, #文档问答
    'apiutil': APIUtil, #API调用
    'apifind': APIFind, #API检索
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLM种类')
    parser.add_argument("--model_name_or_path", type=str, help="模型路径",required=True)
    parser.add_argument('--lora_path', default='', type=str, help='LORA路径')
    parser.add_argument('--eval_data', default='all', type=str, help='评测数据集名称')
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', choices=["False", "True"], default="False", help='only use CPU for inference')
    args = parser.parse_args()
    print(args)
    if args.only_cpu == 'True':
        args.gpus = ""
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(device)
    model_name_or_path = args.model_name_or_path
    model_name = args.model
    lora_path = None if args.lora_path == '' else args.lora_path
    eval_data = args.eval_data
    
    # 加载模型
    llm = model_lists[model_name](model_name_or_path,device,lora_path)

    output_dir = 'output'
    if lora_path is None:
        output_llm_dir = os.path.join(output_dir, model_name )
    else:
        output_llm_dir = os.path.join(output_dir, model_name + '_lora' )
    if not os.path.exists(output_llm_dir):
        os.makedirs(output_llm_dir)
    
    if eval_data != 'all':
        assert eval_data in Eval_datasets
        evaluator = Eval_datasets.get(eval_data)
        result[0] = evaluator().run_evaluation(llm)
        with open(os.path.join(output_llm_dir,evaluator().dataset + '-output.json'), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    else:
        for _, evaluator in Eval_datasets.items():
            result = evaluator().run_evaluation(llm)
            with open(os.path.join(output_llm_dir,evaluator().dataset + '-output.json'), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        
