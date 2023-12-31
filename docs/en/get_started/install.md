# Installation

1. Set up the FinEval environment:

```shell
conda create --name fineval_venv python=3.8
conda activate fineval_venv
```

If you want to customize the PyTorch version or related CUDA version, please refer to [official documentation](https://pytorch.org/get-started/locally) to prepare the Pytorch environment. In the FinEval environment, `pytorch>=1.13` is required.

If you don't use conda, you can ignore this step.

2. Install FinEval:

```shell
git clone https://github.com/SUFE-AIFLM-Lab/FinEval
cd FinEval
pip install -r requirements.txt
```

requirements.txt is as follows:
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

