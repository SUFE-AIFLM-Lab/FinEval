# 快速上手

## 1. 测试Llama-2-7b-hf模型

我们会以测试Llama-2-7b-hf模型的性能为例，带你熟悉FinEval的一些基本功能，默认为`zero-shot`和`answer-only`。

- 运行前确保已经安装了FinEval，本次实验在单张A800显卡上成功运行。更大的参数量，请参考不同模型的推理资源大小，合理选择计算资源。

- FinEval/code文件夹下放置数据集，并命名为data。

- 下载模型权重，Llama-2-7b-hf权重到data的同级目录(FinEval/code文件夹下)

  ```python
  cd FinEval/code
  git clone https://huggingface.co/NousResearch/Llama-2-7b-hf
  ```

  以下为该项目结构目录

  ```
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
  │   ├── eval.py # 基于权重的模型运行文件
  │   ├── eval_chatgpt.py # 基于chatgpt的模型运行文件
  │   ├── run_eval.sh # 基于模型权重的模型配置脚本
  │   ├── subject_mapping.json # 文件配置信息 文件名称和data下文件名称对应
  │   └── run_chatgpt_eval.sh # chatgpt的配置脚本
  ```

- FinEval的评测配置文件以配置.sh脚本为主，使用`run_eval.sh`启动。

- 模型如果一切正常，屏幕上会出现

  ```
  0.0 Inference starts at 2023-07-27_12-06-31 on llama with subject of finance!
  0% 0/58 [00:00<00:00,  2.61s/it]
  ```

注：可以使用`ctrl+c`中断程序执行。运行demo期间，我们来详细讲解本案例中详细内容和参数配置。

## 2. 最终运行结果

最终运行结果如下：

    ```
    Accuracy_subject:
    banking_practitioner_qualification_certificate :  37.57225433526011
    financial_management :  25.0
    economic_law :  26.923076923076923
    certified_management_accountan :  12.5
    auditing :  25.0
    china_actuary :  31.03448275862069
    international_finance :  33.333333333333336
    investments :  29.545454545454547
    central_banking :  52.0
    public_finance :  40.0
    financial_markets :  45.0
    international_economics :  10.0
    finance :  28.0
    intermediate_financial_accounting :  15.384615384615385
    commercial_bank_finance :  30.0
    monetary_finance :  25.58139534883721
    corporate_strategy_and_risk_management :  22.22222222222222
    fund_qualification_certificate :  39.705882352941174
    econometrics :  40.0
    certified_practising_accountant :  23.529411764705884
    insurance :  33.333333333333336
    securities_practitioner_qualification_certificate :  18.181818181818183
    statistics :  25.714285714285715
    advanced_financial_accounting :  26.08695652173913
    financial_engineering :  56.0
    political_economy :  20.0
    microeconomics :  61.111111111111114
    corporate_finance :  32.432432432432435
    tax_law :  37.77777777777778
    cost_accounting :  44.11764705882353
    futures_practitioner_qualification_certificate :  38.46153846153846
    accounting :  19.444444444444443
    management_accounting :  43.18181818181818
    macroeconomics :  30.434782608695652
    --------------------------------------------------------------------------------
    Accuracy_grouped:
    Accounting :  30.434782608695652
    Certificate :  34.120734908136484
    Economy :  32.16374269005848
    Finance :  35.042735042735046
    Avg: 
    33.19467554076539
    ```

## 3. 模型分数解读：

​		1、Accuracy_subject下分数为每个科目的具体分数、Accuracy_grouped为各个科目所属类别的具体分数、Avg为该模型的最终分数（即基于类别总数对四个类别加权平均的结果）

​		2、四选一，所以 baseline 是 25 分，但是模型没训练好的话可能低于 25 分。

​		3、CoT 不一定能显著提升模型分数因为只有在推理数据类任务上，模型强到一定程度之后，CoT 才会有效，这也是为什么 CoT 是一个典型的涌现能力。

​		4、CoT 的模式下，目前只评价最终答案对不对，不评价中间过程对不对，这是因为中间过程和最终答案在大部分时候显著正相关，最终答案对了，中间不会错到哪里去；中间错的多了，最终答案不会对；这种做法可以绕开中间过程难以评价的问题。

​		5、具体的分数的显著性还跟模型天生的 variance 相关，因此推荐多跑实验观察。

