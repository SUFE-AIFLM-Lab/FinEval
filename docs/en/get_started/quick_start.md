# Get started quickly

## 1. Testing the Llama-2-7b-hf model

We will take the performance of the Llama-2-7b-hf model as an example to familiarize you with some basic functions of FinEval, which are `zero-shot` and `answer-only` by default.

- Make sure you have installed FinEval before running. This experiment runs successfully on a single A800 graphics card. For a larger number of parameters, please refer to the inference resource size of different models, and choose computing resources reasonably.

- Place the dataset under the FinEval/code folder and name it data.

- Download model weights, Llama-2-7b-hf weights to the same directory as data (under the FinEval/code folder)

  ```python
  cd FinEval/code
  git clone https://huggingface.co/NousResearch/Llama-2-7b-hf
  ```

  The following is the project structure directory:

  ```
  Fineval/
  ├── requirements
  ├── docs
  ├── README.md
  ├── ...
  ├── code  # Evaluation code
  │   ├── Llama-2-7b-hf  # Model weight (the position of the model can be placed arbitrarily, just change the model_path to the absolute address of the model weight in the run_eval.sh file)
  │   ├── data  # data set
  │	     ├── dev 
  │	     ├── val 
  │	     ├── test
  │   ├── evaluators
  │	     ├── chatgpt.py
  │	     ├── evaluator.py
  │	     ├── unify_evaluator.py
  │   ├── README.md
  │   ├── eval.py # Weight-based model run file
  │   ├── eval_chatgpt.py # Model run file based on chatgpt
  │   ├── run_eval.sh # Model configuration script based on model weights
  │   ├── subject_mapping.json # File configuration information The file name corresponds to the file name under data
  │   └── run_chatgpt_eval.sh # chatgpt configuration script
  ```

- FinEval's evaluation configuration file is mainly based on the configuration.sh script, which is started with `run_eval.sh`.

- The model will appear on the screen if everything is ok

  ```
  0.0 Inference starts at 2023-07-27_12-06-31 on llama with subject of finance!
  0% 0/58 [00:00<00:00,  2.61s/it]
  ```

Note: You can use `ctrl+c` to interrupt program execution. During the running of the demo, let's explain in detail the detailed content and parameter configuration in this case.

## 2. Final running result

The final running result is as follows:

    ```
    Accuracy_subject:
    banking_practitioner_qualification_certificate :  37.57225433526011
    financial_management :  25.0
    economic_law :  26.923076923076923
    certified_management_accountan :  12.5
    auditing :  25.0
    china_actuary :  31.03448275862069
    international_finance :  33.333333333333336
    investments :  29.545454545454547
    central_banking :  52.0
    public_finance :  40.0
    financial_markets :  45.0
    international_economics :  10.0
    finance :  28.0
    intermediate_financial_accounting :  15.384615384615385
    commercial_bank_finance :  30.0
    monetary_finance :  25.58139534883721
    corporate_strategy_and_risk_management :  22.22222222222222
    fund_qualification_certificate :  39.705882352941174
    econometrics :  40.0
    certified_practising_accountant :  23.529411764705884
    insurance :  33.333333333333336
    securities_practitioner_qualification_certificate :  18.181818181818183
    statistics :  25.714285714285715
    advanced_financial_accounting :  26.08695652173913
    financial_engineering :  56.0
    political_economy :  20.0
    microeconomics :  61.111111111111114
    corporate_finance :  32.432432432432435
    tax_law :  37.77777777777778
    cost_accounting :  44.11764705882353
    futures_practitioner_qualification_certificate :  38.46153846153846
    accounting :  19.444444444444443
    management_accounting :  43.18181818181818
    macroeconomics :  30.434782608695652
    --------------------------------------------------------------------------------
    Accuracy_grouped:
    Accounting :  30.434782608695652
    Certificate :  34.120734908136484
    Economy :  32.16374269005848
    Finance :  35.042735042735046
    Avg: 
    33.19467554076539
    ```

## 3. Model score interpretation:

​ 1. The score under Accuracy_subject is the specific score of each subject, Accuracy_grouped is the specific score of the category to which each subject belongs, and Avg is the final score of the model (that is, the weighted average of the four categories based on the total number of categories)

2. Choose one of the four, so the baseline is 25 points, but if the model is not trained well, it may be lower than 25 points.

​ 3. CoT may not be able to significantly improve the model score, because CoT will be effective only after the model is strong enough to a certain extent in reasoning data tasks, which is why CoT is a typical emergent ability.

​ 4. Under the CoT mode, currently only the final answer is evaluated, and the intermediate process is not evaluated. This is because the intermediate process and the final answer are significantly positively correlated most of the time. The final answer is correct, and there will be no mistakes in the middle. Go; if there are too many mistakes in the middle, the final answer will not be right; this approach can bypass the difficult-to-evaluate problems in the middle process.

5. The significance of the specific score is also related to the inherent variance of the model, so it is recommended to run more experiments to observe.
