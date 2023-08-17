# 数据集准备

使用Hugging Face datasets下载数据集。运行命令进行**手动下载解压**，在FinEval/code的项目目录下运行下面命令，数据集准备至FinEval/code/data目录下。

```text
cd code/data
wget https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval/resolve/main/FinEval.zip
unzip FinEval.zip
```

数据集解压后，文件格式如下:

- -----data
  - ----dev：每个科目的dev集中包含五个示范实例以及few-shot评估提供的解释
  - ----val：val集主要用于自测模型效果，可直接得到分数
  - ----test：用于模型评估，test集的答案不会公开，需用户提交测评结果，才可以获得测试准确值
