# Get started quickly

## 1. Testing the Llama-2-7b-hf model

We will take the performance of the Llama-2-7b-hf model as an example to familiarize you with some basic functions of FinEval, which are `zero-shot` and `answer-only` by default.

- Make sure you have installed FinEval before running. This experiment runs successfully on a single A800 graphics card. For a larger number of parameters, please refer to the inference resource size of different models, and choose computing resources reasonably.

- Place the dataset under the FinEval/code folder and name it data.

- Download model weights, Llama-2-7b-hf weights to the same directory as data (under the FinEval/code folder)

```shell
cd FinEval/code
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf
```

  The following is the project structure directory:

```text
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
economic_law :16.0
auditing : 34.375
china_actuary :32.432432432432435international finance :23.529411764705884
investments :28.94736842105263
central_banking : 35.714285714285715
public finance :35.0
financial markets :23.076923076923077
international economics :20.0
finance :36.0
intermediate_financial_accounting : 34.61538461538461
commercial bank finance :5.0
monetary finance :23.25581395348837
corporate_strategy_and_risk_management : 39.39393939393939fund qualification certificate :36.76470588235294
econometrics :44.44444444444444
38.23529411764706certified practising accountant 
insurance 30.303030303030305securities_practitioner_qualification_certificate :31.818181818181817statistics :25.714285714285715
advanced_financial_accounting : 28.571428571428573
financial_engineering : 46.15384615384615political economy :39.130434782608695
microeconomics :35.0
33,333333333333336corporate finance :
tax law :40.日
cost_accounting :58.8235294117647
futures_practitioner_qualification certificate :28.205128205128204accounting : 38.888888888888886
management_accounting : 37.93103448275862
macroeconomics :41.935483870967744
Accuracy_grouped:
38.032786885245905Accounting :33.83233532934132Certificate :34.29951690821256Economy :
Finance :28.852459016393443
Avg:
33.70981754995656
```

## 3. Model score interpretation:

​ 1. The score under Accuracy_subject is the specific score of each subject, Accuracy_grouped is the specific score of the category to which each subject belongs, and Avg is the final score of the model (that is, the weighted average of the four categories based on the total number of categories)

2. Choose one of the four, so the baseline is 25 points, but if the model is not trained well, it may be lower than 25 points.

​ 3. CoT may not be able to significantly improve the model score, because CoT will be effective only after the model is strong enough to a certain extent in reasoning data tasks, which is why CoT is a typical emergent ability.

​ 4. Under the CoT mode, currently only the final answer is evaluated, and the intermediate process is not evaluated. This is because the intermediate process and the final answer are significantly positively correlated most of the time. The final answer is correct, and there will be no mistakes in the middle. Go; if there are too many mistakes in the middle, the final answer will not be right; this approach can bypass the difficult-to-evaluate problems in the middle process.

5. The significance of the specific score is also related to the inherent variance of the model, so it is recommended to run more experiments to observe.
