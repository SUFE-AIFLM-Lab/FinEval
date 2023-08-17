# Dataset Preparation

Download the dataset using Hugging Face datasets. Run the command to **manually download** and decompress, run the following command in the Fineval/code project directory, and rename it to data, and prepare the dataset to the FinEval/code/data directory.

```text
cd code/data
wget https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval/resolve/main/FinEval.zip
unzip FinEval.zip
```

After the dataset is decompressed, the file format is as follows:

- -----data
   ----dev: The dev set for each subject contains five demonstration examples and explanations provided by the few-shot assessment
   ----val: The val set is mainly used for the self-test model score, and the score can be obtained directly
   ----test: used for the final evaluation of the model, the answers of the test set will not be made public, users are required to submit the evaluation results of `submission.json`, and the obtained scores will participate in the final leaderboard

