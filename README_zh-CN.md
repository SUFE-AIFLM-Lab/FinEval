<div align="center">
  <img src="docs/zh_cn/_static/image/FinEval.jpg" width="500px"/>
  <br />
  <br />

[![license](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/InternLM/opencompass/blob/main/LICENSE)

[🌐网站](https://opencompasstest1.readthedocs.io/zh_CN/latest/index.html) |
[🤗Hugging Face](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval) |
[📃论文](https://arxiv.org/abs/2305.08322)

[English](/README.md) | 简体中文
</div>

欢迎来到**FinEval**

大型语言模型（LLMs）在各种自然语言处理任务中表现出色，然而它们在更具挑战性和特定领域任务中的效力仍然很少被探索。本文介绍了FinEval，这是一个专门为LLMs中的金融领域知识设计的基准测试。

FinEval是一个包含**金融、经济、会计和证书**等领域高质量多项选择题的集合。它包括了4,661个问题，涵盖了34个不同的学科。为了确保对模型性能的全面评估，FinEval采用了零样本、少样本、仅答案和链式思维提示等多种方法。在FinEval上评估最先进的中文和英文LLMs，结果显示只有GPT-4在不同的提示设置下达到了70%的准确率，表明LLMs在金融领域知识方面具有显著的增长潜力。我们的工作提供了一个更全面的金融知识评估基准，利用纸质实践题目，涵盖了广泛的LLMs评估范围。

## 目录

- [性能排行榜](#性能排行榜)
  - [仅预测答案](#仅预测答案)
  - [思维链](#思维链)
- [安装](#安装)
- [评测](#评测)
- [准备数据集](#准备数据集)
- [支持新数据集和模型](#支持新数据集和模型)
- [如何提交](#如何提交)
- [引用](#引用)


## 性能排行榜

我们分为**仅预测答案**和**思维链**对模型进行评估，如果需要了解两种方法的Prompt样例，请参考[仅预测答案的zero-shot](/docs/zh_cn/prompt/zero_shot.md)、[仅预测答案的few-shot](/docs/zh_cn/prompt/few_shot.md)和[思维链](/docs/zh_cn/prompt/cot.md)。

下面是模型的zero-shot和five-shot准确率:


### 仅预测答案

#### Zero-shot
| Model               | Finance | Accounting | Economy | Certificate | Average |
| ------------------- | :-----: | :--------: | :-----: | :---------: | :-----: |
| Random              | 25.0    |    25.0    |  25.0   |    25.0     |  25.0  |
| GPT-4               | 65.2 |      74.7      |    62.5    | 64.7  |  **66.4**   |
| GPT-3.5-turbo       | 49.0 |      58.0      |    48.8    | 50.4  |  51.0   |
| Baichuan-7B         | 48.5 |      58.6      |    47.3    | 50.1  |  50.5   |
| Baichuan-13B-base   | 39.1 |      53.0      |    47.7    | 42.7  |  44.3   |
| Baichuan-13B-chat   | 36.7 |      55.8      |    47.7    | 43.0  |  44.0   |
| LLaMA-7B-hf | 38.6 |      47.6      |    39.5    | 39.0  |  40.6   |
| Chinese-Alpaca-Plus-7B    | 33.3 |      48.3      |    41.3    | 38.0  |  38.9   |
| LLaMA-2-7B-base          | 32.6 |      41.2      |    34.1    | 33.0  |  34.7   |
| LLaMA-2-13B-base   | 31.6 |      37.0      |    33.4    | 32.1  |  33.1   |
| LLaMA-2-13B-chat   | 27.4 |      39.2      |    32.5    | 28.0  |  30.9   |
| LLaMA2-70B-chat    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| ChatGLM-6B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| ChatGLM2-6B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Bloomz-7B1    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| InternLM-7B-chat    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Ziya-LLaMA-13B-v1    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Falcon-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Falcon-40B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Aquila-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| AquilaChat-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| moss-moon-003-base    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| moss-moon-003-sft    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |


#### Five-shot
| Model               | Finance | Accounting | Economy | Certificate | Average |
| ------------------- | :-----: | :--------: | :-----: | :---------: | :-----: |
| Random              | 25.0    |    25.0    |  25.0   |    25.0     |  25.0  |
| GPT-4               | 65.2 |      74.7      |    62.5    | 64.7  |  **66.4**   |
| GPT-3.5-turbo       | 49.0 |      58.0      |    48.8    | 50.4  |  51.0   |
| Baichuan-7B         | 48.5 |      58.6      |    47.3    | 50.1  |  50.5   |
| Baichuan-13B-base   | 39.1 |      53.0      |    47.7    | 42.7  |  44.3   |
| Baichuan-13B-chat   | 36.7 |      55.8      |    47.7    | 43.0  |  44.0   |
| LLaMA-7B-hf | 38.6 |      47.6      |    39.5    | 39.0  |  40.6   |
| Chinese-Alpaca-Plus-7B    | 33.3 |      48.3      |    41.3    | 38.0  |  38.9   |
| LLaMA-2-7B-base          | 32.6 |      41.2      |    34.1    | 33.0  |  34.7   |
| LLaMA-2-13B-base   | 31.6 |      37.0      |    33.4    | 32.1  |  33.1   |
| LLaMA-2-13B-chat   | 27.4 |      39.2      |    32.5    | 28.0  |  30.9   |
| LLaMA2-70B-chat    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| ChatGLM-6B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| ChatGLM2-6B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Bloomz-7B1    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| InternLM-7B-chat    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Ziya-LLaMA-13B-v1    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Falcon-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Falcon-40B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Aquila-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| AquilaChat-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| moss-moon-003-base    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| moss-moon-003-sft    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |


### 思维链

#### Zero-shot
| Model               | Finance | Accounting | Economy | Certificate | Average |
| ------------------- | :-----: | :--------: | :-----: | :---------: | :-----: |
| Random              | 25.0    |    25.0    |  25.0   |    25.0     |  25.0  |
| GPT-4               | 65.2 |      74.7      |    62.5    | 64.7  |  **66.4**   |
| GPT-3.5-turbo       | 49.0 |      58.0      |    48.8    | 50.4  |  51.0   |
| Baichuan-7B         | 48.5 |      58.6      |    47.3    | 50.1  |  50.5   |
| Baichuan-13B-base   | 39.1 |      53.0      |    47.7    | 42.7  |  44.3   |
| Baichuan-13B-chat   | 36.7 |      55.8      |    47.7    | 43.0  |  44.0   |
| LLaMA-7B-hf | 38.6 |      47.6      |    39.5    | 39.0  |  40.6   |
| Chinese-Alpaca-Plus-7B    | 33.3 |      48.3      |    41.3    | 38.0  |  38.9   |
| LLaMA-2-7B-base          | 32.6 |      41.2      |    34.1    | 33.0  |  34.7   |
| LLaMA-2-13B-base   | 31.6 |      37.0      |    33.4    | 32.1  |  33.1   |
| LLaMA-2-13B-chat   | 27.4 |      39.2      |    32.5    | 28.0  |  30.9   |
| LLaMA2-70B-chat    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| ChatGLM-6B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| ChatGLM2-6B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Bloomz-7B1    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| InternLM-7B-chat    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Ziya-LLaMA-13B-v1    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Falcon-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Falcon-40B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Aquila-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| AquilaChat-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| moss-moon-003-base    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| moss-moon-003-sft    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |


#### Five-shot
| Model               | Finance | Accounting | Economy | Certificate | Average |
| ------------------- | :-----: | :--------: | :-----: | :---------: | :-----: |
| Random              | 25.0    |    25.0    |  25.0   |    25.0     |  25.0  |
| GPT-4               | 65.2 |      74.7      |    62.5    | 64.7  |  **66.4**   |
| GPT-3.5-turbo       | 49.0 |      58.0      |    48.8    | 50.4  |  51.0   |
| Baichuan-7B         | 48.5 |      58.6      |    47.3    | 50.1  |  50.5   |
| Baichuan-13B-base   | 39.1 |      53.0      |    47.7    | 42.7  |  44.3   |
| Baichuan-13B-chat   | 36.7 |      55.8      |    47.7    | 43.0  |  44.0   |
| LLaMA-7B-hf | 38.6 |      47.6      |    39.5    | 39.0  |  40.6   |
| Chinese-Alpaca-Plus-7B    | 33.3 |      48.3      |    41.3    | 38.0  |  38.9   |
| LLaMA-2-7B-base          | 32.6 |      41.2      |    34.1    | 33.0  |  34.7   |
| LLaMA-2-13B-base   | 31.6 |      37.0      |    33.4    | 32.1  |  33.1   |
| LLaMA-2-13B-chat   | 27.4 |      39.2      |    32.5    | 28.0  |  30.9   |
| LLaMA2-70B-chat    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| ChatGLM-6B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| ChatGLM2-6B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Bloomz-7B1    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| InternLM-7B-chat    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Ziya-LLaMA-13B-v1    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Falcon-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Falcon-40B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| Aquila-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| AquilaChat-7B    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| moss-moon-003-base    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |
| moss-moon-003-sft    | 28.8 |      32.9      |    29.7    | 28.0  |  29.6   |


## 安装

下面展示了快速安装的步骤，详细请参考[安装指南](docs/zh_cn/get_started/install.md)。

 ```python
    conda create --name fineval_venv python=3.8
    conda activate fineval_venv
 ```

```python
    git clone https://github.com/SUFE-AIFLM/FinEval
    cd FinEval
    pip install -r requirements.txt
    
    requirements.txt 文件如下:
    pandas
    torch
    tqdm
    peft 
    sentencepiece
```

## 准备数据集

使用Hugging Face datasets下载数据集。运行命令进行**手动下载解压**，在Fineval/code的项目目录下运行下面命令，并改名为data，数据集准备至FinEval/code/data目录下。

```
cd code
git clone *----------------
unzip xx.zip
mv xx data
```

data文件夹格式为:

- -----data
  - ----dev：每个科目的dev集中包含五个示范实例以及few-shot评估提供的解释
  - ----val：val集主要作用于超参调整
  - ----test：用于模型评估，test集的标签不会公开，需用户提交其结果，才可以获得测试准确值

## 评测

请阅读[快速上手](/docs/zh_cn/get_started/quick_start.md)了解如何运行一个评测任务。

## 支持新数据集和模型

如果需要新加入数据集进行评测，请参考[支持新数据集](/docs/zh_cn/advanced_guide/new_dataset.md)。

如果需要加载新模型，请参考[支持新模型](/docs/zh_cn/advanced_guide/new_model.md)。

## 如何提交

您首先需要准备一个UTF-8编码的JSON文件，并按照以下格式编写。
```
## 每个学科内部的键名是数据集中的"id"字段
{
    "banking_practitioner_qualification_certificate": {
        "0": "A",
        "1": "B",
        "2": "B",
        ...
    },
    
    "学科名称":{
    "0":"答案1",
    "1":"答案2",
    ...
    }
    ....
}
```
然后你可以将准备好的JSON文件提交到zhang.liwen@shufe.edu.cn。

## 引用

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/InternLM/OpenCompass}},
    year={2023}
}
```
