# Dataset Preparation

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

