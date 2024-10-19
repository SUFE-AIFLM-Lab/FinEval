# 快速上手

## 1. 测试Llama-2-7b-hf模型

我们会以测评Llama-2-7b-hf模型为例，带你熟悉FinEval的一些基本功能，默认为`zero-shot`和`answer-only`。

- 运行前确保已经安装了FinEval，本次实验在单张A800显卡上成功运行。更大的参数量，请参考不同模型的推理资源大小，合理选择计算资源。

- FinEval/code文件夹下放置数据集，并命名为data。

- 下载模型权重，Llama-2-7b-hf权重到FinEval/code文件夹下(与data同级)。

```text
cd FinEval/code
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf
```

  以下为该项目结构目录

```text
Fineval/
├── requirements
├── docs
├── README.md
├── ...
├── code  # 评测代码
│   ├── Llama-2-7b-hf  # 模型权重（模型位置可以任意放置，在run_eval.sh文件中model_path更改为模型权重绝对地址即可）
│   ├── data  # 数据集
│	     ├── dev 
│	     ├── val 
│	     ├── test
│   ├── evaluators
│	     ├── chatgpt.py
│	     ├── evaluator.py
│	     ├── unify_evaluator.py
│   ├── README.md
│   ├── eval.py # 基于本地模型权重的模型运行文件
│   ├── eval_chatgpt.py # 基于chatgpt的模型运行文件
│   ├── run_eval.sh # 基于本地模型权重的测评脚本
│   ├── subject_mapping.json # 数据集配置文件
│   └── run_chatgpt_eval.sh # chatgpt的测评脚本
```

- FinEval的评测配置文件以配置.sh脚本为主，使用`run_eval.sh`启动。

- 模型如果一切正常，屏幕上会出现

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

注：可以使用`ctrl+c`中断程序执行。运行测评期间，屏幕上会打印题目信息。

## 2. 最终运行结果

最终运行结果如下：

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

## 3. 测评分数解读：

​		1、Accuracy_subject下分数为每个科目的具体分数、Accuracy_grouped为各个科目所属类别的具体分数、Avg为该模型的最终分数（即基于类别总数对四个类别加权平均的结果）

​		2、四选一，所以 baseline 是 25 分，但是模型没训练好的话可能低于 25 分。

​		3、CoT 不一定能显著提升模型分数因为只有在推理数据类任务上，模型强到一定程度之后，CoT 才会有效，这也是为什么 CoT 是一个典型的涌现能力。

​		4、CoT 的模式下，目前只评价最终答案对不对，不评价中间过程对不对，这是因为中间过程和最终答案在大部分时候显著正相关，最终答案对了，中间不会错到哪里去；中间错的多了，最终答案不会对；这种做法可以绕开中间过程难以评价的问题。

​		5、具体的分数的显著性还跟模型天生的 variance 相关，因此推荐多跑实验观察。

