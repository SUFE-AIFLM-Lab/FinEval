# Get started quickly

## 1. Testing the Llama-2-7b-hf model

We will take the evaluation of the Llama-2-7b-hf model as an example to familiarize you with some basic functions of FinEval. The defaults are `zero-shot` and `answer-only`.

- Make sure that FinEval has been installed before running. This experiment runs successfully on a single A800 graphics card. For a larger number of parameters, please refer to the size of inference resources of different models, and choose computing resources reasonably.

- Place the dataset under the FinEval/code folder and name it data.

- Download model weights, Llama-2-7b-hf weights to the FinEval/code folder (same level as data).

  
```shell
cd FinEval/code
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf
```

  The following is the project structure directory:

```text
Fineval/
├── requirements
├── docs
├── README.md
├── ...
├── code  # Evaluation code
│   ├── Llama-2-7b-hf  # Model weight (the position of the model can be placed arbitrarily, just change the model_path to the absolute address of the model weight in the run_eval.sh file)
│   ├── data  # data set
│	     ├── dev 
│	     ├── val 
│	     ├── test
│   ├── evaluators
│	     ├── chatgpt.py
│	     ├── evaluator.py
│	     ├── unify_evaluator.py
│   ├── README.md
│   ├── eval.py # Model run file based on local model weights
│   ├── eval_chatgpt.py # Model run file based on chatgpt
│   ├── run_eval.sh # Evaluation script based on local model weights
│   ├── subject_mapping.json # Dataset configuration file
│   └── run_chatgpt_eval.sh # chatgpt evaluation script
```

- FinEval's evaluation configuration file is mainly based on the configuration.sh script, which is started with `run_eval.sh`.

- The model will appear on the screen if everything is ok

```text
exp_date: 时间戳
output_path: your output_path
[2023-08-17 22:37:28,483] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Namespace(constrained_decoding=True, cot=False, do_save_csv=True, do_test=False, few_shot=False, gpus='0,1', lora_model='', \
model_path=your model_path, model_type='llama', n_times=1, ntrain=5, only_cpu='False', output_dir=your output_path, temperature=0.2, with_prompt=False)
cuda:0
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████2/2 [00:02<00:00,  1.28s/it]
0.0 Inference starts at exp_date on your model_path with subject of banking_practitioner_qualification_certificate!
  0%|                                                                                                                                                                                                            | 0/116 [00:00<?, ?it/s]
```

Note: You can use `ctrl+c` to interrupt program execution. While the assessment is running, question information is printed on the screen.

## 2. Final running result

The final running result is as follows:

```text
Accuracy_subject:
banking_practitioner_qualification_certificate :  44.827586206896555
certified_management_accountant :  27.77777777777778
financial_management :  25.0
economic_law :  32.0
auditing :  25.0
china_actuary :  21.62162162162162
international_finance :  41.1764705882353
investments :  28.94736842105263
central_banking :  50.0
public_finance :  47.5
financial_markets :  43.58974358974359
international_economics :  20.0
finance :  24.0
intermediate_financial_accounting :  15.384615384615385
commercial_bank_finance :  35.0
monetary_finance :  25.58139534883721
corporate_strategy_and_risk_management :  12.121212121212121
fund_qualification_certificate :  41.1764705882353
econometrics :  33.333333333333336
certified_practising_accountant :  23.529411764705884
insurance :  36.36363636363637
securities_practitioner_qualification_certificate :  22.727272727272727
statistics :  22.857142857142858
advanced_financial_accounting :  19.047619047619047
financial_engineering :  53.84615384615385
political_economy :  21.73913043478261
microeconomics :  27.5
corporate_finance :  38.888888888888886
tax_law :  37.77777777777778
cost_accounting :  32.35294117647059
futures_practitioner_qualification_certificate :  38.46153846153846
accounting :  19.444444444444443
management_accounting :  44.827586206896555
macroeconomics :  19.35483870967742
--------------------------------------------------------------------------------
Accuracy_grouped:
Accounting :  26.885245901639344
Certificate :  36.22754491017964
Economy :  28.502415458937197
Finance :  37.049180327868854
Avg: 
32.580364900086884
```

## 3. Interpretation of test scores

​1. The score under Accuracy_subject is the specific score of each subject, Accuracy_grouped is the specific score of the category to which each subject belongs, and Avg is the final score of the model (that is, the weighted average of the four categories based on the total number of categories)

2. Choose one of the four, so the baseline is 25 points, but if the model is not trained well, it may be lower than 25 points.

3. CoT may not be able to significantly improve the model score, because CoT will be effective only after the model is strong to a certain extent in inference data tasks, which is why CoT is a typical emergent ability.

​4. In the CoT mode, currently only the final answer is evaluated, and the intermediate process is not evaluated. This is because the intermediate process and the final answer are significantly positively correlated most of the time. The final answer is correct, and there will be no mistakes in the middle. Where to go; if there are too many mistakes in the middle, the final answer will not be right; this approach can bypass the difficult-to-evaluate problems in the middle process.

5. The significance of the specific score is also related to the inherent variance of the model, so it is recommended to run more experiments to observe.
