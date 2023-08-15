<div align="center">
  <img src="docs/zh_cn/_static/image/FinEval.jpg" width="500px"/>
  <br />
  <br />

[![license](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/SUFE-AIFLM-Lab/FinEval/blob/main/LICENSE)

[🌐网站](https://fineval.readthedocs.io/zh_CN/latest/) |
[🤗Hugging Face](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval) |
[📃论文]()

[English](/README.md) | 简体中文
</div>

欢迎来到**FinEval**

大型语言模型（LLMs）在各种自然语言处理任务中展现出卓越的性能，然而它们在更具挑战性和领域特定的任务中的功效仍然未被充分探索。本文介绍了FinEval，一个专门为LLMs中的金融领域知识而设计的基准测试。

FinEval是一个包含高质量多项选择题的集合，涵盖**金融、经济、会计和证书**等领域。它包括4,661个问题，涵盖了34个不同的学术科目。为了确保对模型性能进行全面的评估，FinEval采用了多种方法，包括zero-shot，few-shot，仅预测答案（answer-only）和思维链（chain-of-thought）提示词。通过在FinEval上评估最先进的中英文大语言模型，结果显示只有GPT-4在不同提示设置下达到了60%的准确率，表明大语言模型在金融领域知识方面具有显著的增长潜力。我们的工作提供了一个更全面的金融知识评估基准，利用了模拟考试数据，涵盖了广泛的大语言模型评估范围。

<div align="center">
  <img src="docs/en/_static/image/subjects.png" width="1000px" height="475px"/>
  <br />
  <br /></div>

## 目录

- [性能排行榜](#性能排行榜)
- [安装](#安装)
- [评测](#评测)
- [准备数据集](#准备数据集)
- [支持新数据集和模型](#支持新数据集和模型)
- [如何提交](#如何提交)
- [引用](#引用)


## 性能排行榜

我们分为**仅预测答案**和**思维链**对模型进行评估，如果需要了解两种方法的Prompt样例，请参考[仅预测答案的zero-shot](/docs/zh_cn/prompt/zero_shot.md)、[仅预测答案的few-shot](/docs/zh_cn/prompt/few_shot.md)和[思维链](/docs/zh_cn/prompt/cot.md)。

下面是我们评估模型在测试集（test）上的平均准确率(%)。每个类别下的平均准确率是该类别下所有学科的平均准确率，最后一列是模型在所有学科上的平均准确率。此外，在四种Prompt设置下，我们只给出了所有学科平均准确率最高的设置结果。

| Model                  | Size    | Finance | Economy | Accounting | Certificate | Average |
|------------------------|---------|:-------:|:-------:|:----------:|:-----------:|:-------:|
| GPT-4                  | unknown |  71.0   |  74.5   |    59.3    |    70.4     |  68.6   |
| ChatGPT                | 175B    |  59.3   |  61.6   |    45.2    |    55.1     |  55.0   |
| Qwen-7B                | 7B      |  54.5   |  54.4   |    50.3    |    55.8     |  53.8   |
| Qwen-Chat-7B           | 7B      |  51.5   |  52.1   |    44.5    |    53.6     |  50.5   |
| Baichuan-13B-Base      | 13B     |  52.6   |  50.2   |    43.4    |    53.5     |  50.1   |
| Baichuan-13B-Chat      | 13B     |  51.6   |  51.1   |    41.7    |    52.8     |  49.4   |
| ChatGLM2-6B            | 6B      |  46.5   |  46.4   |    44.5    |    51.5     |  47.4   |
| InternLM-7B            | 7B      |  49.0   |  49.2   |    40.5    |    49.4     |  47.1   |
| InternLM-Chat-7B       | 7B      |  48.4   |  49.1   |    40.8    |    49.5     |  47.0   |
| LLaMA-2-Chat-70B       | 70B     |  47.1   |  46.7   |    41.5    |    45.7     |  45.2   |
| Falcon-40B             | 40B     |  45.4   |  43.2   |    35.8    |    44.8     |  42.4   |
| Baichuan-7B            | 7B      |  44.9   |  41.5   |    34.9    |    45.6     |  42.0   |
| LLaMA-2-Chat-13B       | 13B     |  41.6   |  38.4   |    34.1    |    42.1     |  39.3   |
| Ziya-LLaMA-13B-v1      | 13B     |  43.3   |  36.9   |    34.3    |    41.2     |  39.3   |
| Bloomz-7b1-mt          | 7B      |  41.4   |  42.1   |    32.5    |    39.7     |  38.8   |
| LLaMA-2-13B            | 13B     |  39.5   |  38.6   |    31.6    |    39.6     |  37.4   |
| ChatGLM-6B             | 6B      |  38.8   |  36.2   |    33.8    |    39.1     |  37.2   |
| Chinese-Llama-2-7B     | 7B      |  37.8   |  37.8   |    31.4    |    36.7     |  35.9   |
| Chinese-Alpaca-Plus-7B | 7B      |  30.5   |  33.4   |    32.7    |    38.5     |  34.0   |
| moss-moon-003-sft      | 16B     |  35.6   |  34.3   |    28.7    |    35.6     |  33.7   |
| LLaMA-2-Chat-7B        | 7B      |  35.6   |  31.8   |    31.9    |    34.0     |  33.5   |
| LLaMA-2-7B             | 7B      |  34.9   |  36.4   |    31.4    |    31.6     |  33.4   |
| AquilaChat-7B          | 7B      |  34.2   |  31.3   |    29.8    |    36.2     |  33.1   |
| moss-moon-003-base     | 16B     |  32.2   |  33.1   |    29.2    |    30.7     |  31.2   |
| Aquila-7B              | 7B      |  27.1   |  31.6   |    32.4    |    33.6     |  31.2   |
| LLaMA-13B              | 13B     |  33.1   |  29.7   |    27.2    |    33.6     |  31.1   |
| Falcon-7B              | 7B      |  28.5   |  28.2   |    27.5    |    27.4     |  27.9   |


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
    numpy
    torch
    tqdm
    peft 
    sentencepiece
    openai
    accelerate
    colorama
    cpm_kernels
    sentencepiece
    streamlit
    transformers_stream_generator
    transformers==4.31.0
    tiktoken
    einops
    scipy
```

## 准备数据集

使用Hugging Face datasets下载数据集。运行命令进行**手动下载解压**，在Fineval/code的项目目录下运行下面命令，并改名为data，数据集准备至FinEval/code/data目录下。

```
cd code/data
wget https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval/resolve/main/FinEval.zip
unzip FinEval.zip
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
