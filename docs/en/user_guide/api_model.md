# API-based Models

Taking OpenAi's api model as an example, use the `code/run_chatgpt_eval.sh` script file to run the evaluation. Except for the openai_key parameter, other parameters are consistent with the evaluation parameters based on model weight, and the core parameters are model_name and openai_key.

In the model_name field, please fill in the correct model name. If evaluating gpt4, please fill in the specifications. In the final evaluation script, please fill in gpt-4. For the correct model information, please refer to the OpenAI official website.

In addition, here openai_key recommends using the paid version, and the $5 version has a speed limit that affects the evaluation.

```python
export PROJ_HOME=$PWD
export KMP_DUPLICATE_LIB_OK=TRUE

# Determine the key of the api
openai_key=sk-***************** #Fill in your openai_key here for evaluation

exp_name=chatgpt
exp_date=$(date +"%Y%m%d%H%M%S")
output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date

echo "exp_date": $exp_date
echo "output_path": $output_path

python eval_chatgpt.py \
    --openai_key ${openai_key} \
    --cot False \
    --few_shot False \
    --n_times 1 \
    --ntrain 5 \
    --do_test False \
    --do_save_csv False \
    --output_dir ${output_path} \
    --model_name gpt-4 # Please fill in the correct OpenAI model name
```
