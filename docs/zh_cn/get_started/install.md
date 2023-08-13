# 安装

1. 准备 FinEval 运行环境：

    ```python
    conda create --name fineval_venv python=3.8
    conda activate fineval_venv
    ```

    如果你希望自定义PyTorch版本或者相关CUDA版本，请参考官方文档准备Pytorch环境。FinEval环境中，要求`pytorch>=1.13`。

2. 安装 FinEval：

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
    
