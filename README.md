<div align="center">
  <img src="docs/en/_static/image/FinEval.jpg" width="500px"/>
  <br />
  <br />

[![license](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/SUFE-AIFLM-Lab/FinEval/blob/main/LICENSE)

[🌐Website](https://opencompasstest1.readthedocs.io/zh_CN/latest/index.html) |
[🤗Hugging Face](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval) |
[📃Paper]()

English | [简体中文](README_zh-CN.md)

</div>


Welcome to **FinEval**

Large Language Models (LLMs) have demonstrated impressive performance across various natural language processing tasks. However, their effectiveness in more challenging and domain-specific tasks remains largely unexplored. This article introduces FinEval, a benchmark designed specifically for assessing financial domain knowledge within LLMs.

FinEval comprises a collection of high-quality multiple-choice questions spanning the fields of **finance, economics, accounting, and certifications**. It encompasses 4,661 questions, covering 34 distinct disciplines. To ensure a comprehensive evaluation of model performance, FinEval employs various methods such as zero-shot, few-shot, answer-only, and chain of thought prompts. Evaluating state-of-the-art Chinese and English LLMs on FinEval reveals that only GPT-4 achieves a 70% accuracy rate across different prompt settings, underscoring the significant growth potential of LLMs in financial domain knowledge. Our work provides a more comprehensive benchmark for evaluating financial knowledge, utilizing practical paper-based exercises that encompass a wide range of LLMs assessment scenarios.


## Contents

- [Performance Leaderboard](#Performance-leaderboard)
  - [Answer Only](#Answer-only)
  - [Chain of thought](#chain-of-thought)
- [Installation](#Installation)
- [Evaluation](#Evaluation)
- [Dataset Preparation](#Dataset-Preparation)
- [Supporting New Datasets and Models](#supporting-new-datasets-and-models)
- [How to Submit](#How-to-submit)
- [Citation](#Citation)

## Performance Leaderboard

We divide the evaluation into **Answer Only** and **Chain of Thought**. For examples of prompts for both methods, please refer to [zero-shot for Answer Only](/docs/en/prompt/zero_shot.md), [few-shot for Answer Only](/docs/en/prompt/few_shot.md), and [Chain of Thought](/docs/en/prompt/cot.md).

下面是模型的zero-shot和five-shot准确率:


### Answer Only

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


### Chain of thought

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


## Installation

Below are the steps for quick installation. For detailed instructions, please refer to the [Installation Guide](docs/en/get_started/install.md).

 ```python
    conda create --name fineval_venv python=3.8
    conda activate fineval_venv
 ```

```python
    git clone https://github.com/SUFE-AIFLM/FinEval
    cd FinEval
    pip install -r requirements.txt
    
    requirements.txt is as follows:
    pandas
    torch
    tqdm
    peft 
    sentencepiece
```

## Dataset Preparation

Download the dataset using Hugging Face datasets. Run the command to **manually download** and decompress, run the following command in the Fineval/code project directory, and rename it to data, and prepare the dataset to the FinEval/code/data directory.

```
cd code
git clone *----------------
unzip xx.zip
mv xx data
```

The format of the data folder is:
- -----data
  - ----dev: The dev set for each subject contains five demonstration examples with explanations provided by the few-shot evaluation
  - ----val: The val set is mainly used for hyperparameter adjustment
  - ----test: Used for model evaluation, the labels of the test set will not be disclosed, and users need to submit their results to obtain the accurate value of the test


## Evaluation

Please read [Get started quickly](/docs/en/get_started/quick_start.md) to learn how to run an evaluation task.

## Supporting New Datasets and Models

If you need to incorporate a new dataset for evaluation, please refer to [Add a dataset](/docs/zh_cn/advanced_guides/new_dataset.md).

If you need to load a new model, please refer to [Add a Model](/docs/zh_cn/advanced_guides/new_model.md).

## How to Submit
First, you need to prepare a JSON file encoded in UTF-8 and follow the format below:
```
## The keys within each subject correspond to the "id" field in the dataset
{
    "banking_practitioner_qualification_certificate": {
        "0": "A",
        "1": "B",
        "2": "B",
        ...
    },
    
    "Subject Name":{
    "0":"Answer1",
    "1":"Answer2",
    ...
    }
    ....
}
```
Once you have prepared the JSON file, you can submit it to zhang.liwen@shufe.edu.cn.

## Citation

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/InternLM/OpenCompass}},
    year={2023}
}
```
