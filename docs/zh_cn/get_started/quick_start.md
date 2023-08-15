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
economic_law :16.0
auditing : 34.375
china_actuary :32.432432432432435international finance :23.529411764705884
investments :28.94736842105263
central_banking : 35.714285714285715
public finance :35.0
financial markets :23.076923076923077
international economics :20.0
finance :36.0
intermediate_financial_accounting : 34.61538461538461
commercial bank finance :5.0
monetary finance :23.25581395348837
corporate_strategy_and_risk_management : 39.39393939393939fund qualification certificate :36.76470588235294
econometrics :44.44444444444444
38.23529411764706certified practising accountant 
insurance 30.303030303030305securities_practitioner_qualification_certificate :31.818181818181817statistics :25.714285714285715
advanced_financial_accounting : 28.571428571428573
financial_engineering : 46.15384615384615political economy :39.130434782608695
microeconomics :35.0
33,333333333333336corporate finance :
tax law :40.日
cost_accounting :58.8235294117647
futures_practitioner_qualification certificate :28.205128205128204accounting : 38.888888888888886
management_accounting : 37.93103448275862
macroeconomics :41.935483870967744
Accuracy_grouped:
38.032786885245905Accounting :33.83233532934132Certificate :34.29951690821256Economy :
Finance :28.852459016393443
Avg:
33.70981754995656
```

## 3. 模型分数解读：

​		1、Accuracy_subject下分数为每个科目的具体分数、Accuracy_grouped为各个科目所属类别的具体分数、Avg为该模型的最终分数（即基于类别总数对四个类别加权平均的结果）

​		2、四选一，所以 baseline 是 25 分，但是模型没训练好的话可能低于 25 分。

​		3、CoT 不一定能显著提升模型分数因为只有在推理数据类任务上，模型强到一定程度之后，CoT 才会有效，这也是为什么 CoT 是一个典型的涌现能力。

​		4、CoT 的模式下，目前只评价最终答案对不对，不评价中间过程对不对，这是因为中间过程和最终答案在大部分时候显著正相关，最终答案对了，中间不会错到哪里去；中间错的多了，最终答案不会对；这种做法可以绕开中间过程难以评价的问题。

​		5、具体的分数的显著性还跟模型天生的 variance 相关，因此推荐多跑实验观察。

