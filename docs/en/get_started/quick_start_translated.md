
# Quick Start
We will introduce the usage of FinEval for academic financial knowledge evaluation, industry financial knowledge evaluation, and the financial security knowledge and financial agents open-source evaluation code.

You can find the code for all three parts in the [code/opensource_eval](/code/opensource_eval) folder of this project, divided into academic_eval, industry_eval, and security+agenteval.

## 1. Academic Financial Knowledge Evaluation
### Evaluating the Llama-2-7b-hf Model
We will take the evaluation of the Llama-2-7b-hf model as an example to help you become familiar with some basic FinEval functionalities, which default to `zero-shot` and `answer-only`.

- Before running, ensure FinEval is installed. This experiment was successfully run on a single A800 GPU. For models with larger parameter sizes, refer to the respective model's inference resource requirements and choose computation resources accordingly.

- Place the dataset in the 1academic_eval/code folder and name it `data`.

- Download the model weights (Llama-2-7b-hf) into the 1academic_eval/code folder (at the same level as the `data` folder).

```bash
cd 1academic_eval/code
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf
```

The directory structure for 1academic_eval is as follows:

```text
1academic_eval/
├── requirements
├── docs
├── README.md
├── ...
├── code  # Evaluation code
│   ├── Llama-2-7b-hf  # Model weights (The model location can be placed anywhere, just change the model_path to the absolute path in the run_eval.sh file)
│   ├── data  # Dataset
│        ├── dev 
│        ├── val 
│        ├── test
│   ├── evaluators
│        ├── chatgpt.py
│        ├── evaluator.py
│        ├── unify_evaluator.py
│   ├── README.md
│   ├── eval.py # Model running script based on local weights
│   ├── eval_chatgpt.py # Model running script based on ChatGPT
│   ├── run_eval.sh # Evaluation script for local model weights
│   ├── subject_mapping.json # Dataset configuration file
│   └── run_chatgpt_eval.sh # ChatGPT evaluation script
```

- The evaluation configuration file in 1academic_eval primarily consists of a `.sh` script and is started using `run_eval.sh`.
Example script:

```bash
xuanyuan3_70b=/Llama3-XuanYuan3-70B-Chat

python eval.py --with_prompt False --ntrain 5 --temperature 0.1 --n_times 1 --do_save_csv True --only_cpu False \
    --model_type  ${model_type} \
    --model_path ${xuanyuan3_70b} \
    --cot False \
    --few_shot False \
    --do_test True \
    --gpus 0,1 \
    --output_dir ${output_path}\
```

- If the model runs correctly, the screen will display:

```text
exp_date: timestamp
output_path: your output_path
[2023-08-17 22:37:28,483] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Namespace(constrained_decoding=True, cot=False, do_save_csv=True, do_test=False, few_shot=False, gpus='0,1', lora_model='', \
model_path=your model_path, model_type='llama', n_times=1, ntrain=5, only_cpu='False'...
```

## 2. Industry Financial Knowledge Evaluation
The directory structure for 2industry_eval is as follows:

```text
├── data
│   ├── fincqa-eval.jsonl
│   ├── fincustomer-eval.jsonl
│   ├── finfeforum-eval.jsonl
│   ├── ...
├── eval.py # Model running script based on local weights
├── eval.sh # Evaluation script based on local weights
├── evaluate.py # Dataset evaluation script
├── finllm.py
├── ...
├── requirements.txt
└── utils.py
```

- The evaluation configuration file in 2industry_eval consists of `.sh` scripts, which are started using `eval.sh`.
Example script:

```bash
qwen2_7b=qwen2-7b

python eval.py --model qwen2-7b --model_name_or_path ${qwen2_7b} --gpus 0,1 --eval_data all
python eval.py --model qwen2-7b --model_name_or_path ${qwen2_7b} --gpus 0,1 --eval_data all --cot
```

## 3. Financial Security Knowledge and Financial Agent Evaluation
The directory structure for 34security+agenteval is as follows:

```text
├── data
│   ├── apifind-eval.json
│   ├── apiutil-eval.json
│   ├── appsafe-eval.json
│   ├── crypsafe-eval.json
│   ├── ...
├── eval_agent.py # Agent model running script (using local large model for answers)
├── eval_security.py # Security model running script
├── eval.sh # Evaluation script
├── evaluate.py # Dataset evaluation script
├── finllm.py
├── gpt4eval.py # Agent model running script (using API for scoring)
├── gpt4eval.sh # Agent evaluation script
├── security
│   ├── cryptography_val.csv
│   ├── malwareannalysis_val.csv
│   ├── ...
└── utils.py
```

- The evaluation configuration file in 34security+agenteval primarily consists of `.sh` scripts.
To perform financial security knowledge evaluation, run the `eval.sh` script.
Example script:

```bash
xuanyuan3_70b=llama3-xuanyuan3-70b-chat

python eval_security.py --model xuanyuan3-70b-chat --model_name_or_path $xuanyuan3_70b --gpus 2,3 --eval_data all 
python eval_security.py --model xuanyuan3-70b-chat --model_name_or_path $xuanyuan3_70b --gpus 2,3 --eval_data all --cot
```

- To perform financial agent evaluation, first run the `eval.sh` script, then run the `gpt4eval.sh` script.
Example script:

In `eval.sh`:

```bash
python eval_agent.py --model xuanyuan3-70b-chat --model_name_or_path $xuanyuan3_70b --gpus 2,3 --eval_data all
```

In `gpt4eval.sh`:

```bash
python gpt4eval.py --model xuanyuan3-70b-chat --model_name_or_path ... --gpus 1 --eval_data all
```
