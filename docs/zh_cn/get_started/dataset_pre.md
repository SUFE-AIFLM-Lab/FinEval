# 数据集准备

使用Hugging Face datasets下载数据集。运行命令进行**手动下载解压**，在Fineval/code的项目目录下运行下面命令，并改名为data，数据集准备至FinEval/code/data目录下。

```text
cd code/data
wget https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval/resolve/main/FinEval.zip
unzip FinEval.zip
```

data文件夹格式为:

- -----data
  - ----dev：每个科目的dev集中包含五个示范实例以及few-shot评估提供的解释
  - ----val：val集主要作用于超参调整
  - ----test：用于模型评估，test集的标签不会公开，需用户提交其结果，才可以获得测试准确值
