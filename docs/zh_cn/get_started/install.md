# 安装

1. 准备 FinEval 运行环境：

```text
conda create --name fineval_venv python=3.8
conda activate fineval_venv
```

如果你希望自定义PyTorch版本或者相关CUDA版本，请参考[官方文档](https://pytorch.org/get-started/locally)准备Pytorch环境。FinEval环境中，要求`pytorch>=1.13`。

如果你不使用conda，可以忽略本步骤。

2. 安装 FinEval：

```text
git clone https://github.com/SUFE-AIFLM-Lab/FinEval
cd FinEval
pip install -r requirements.txt
```

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

    
