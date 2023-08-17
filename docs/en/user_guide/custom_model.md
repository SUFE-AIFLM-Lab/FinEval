# Custom model

When defining a new model, define the model type as auto to load the new model. You can change other parameters by yourself. Here, the Baichuan model is taken as an example to load a custom model.

If the newly added model parameters are configured as AutoModelForCausalLM, AutoTokenizer loading, and the model type is auto, it can be evaluated.
Here is a code sample of `run_eval.sh`:

```text
#!/bin/bash
#/*******
#1, increase the choice of CPU loading model, parameterized configuration
#2. Increase the type selection of the model and parameterize the configuration
#3, support lora weight loading, parameterized configuration
#4. Support the setting of the number of GPU loading

#baichuan-13b
model_type=auto #model_type=auto #If the model type does not exist, you can use the auto method to load, and use the AutoModelForCausalLM, AutoTokenizer method to load
model_path=/baichuan-13b
exp_name=baichuan13b

exp_date=$(date +"%Y%m%d%H%M%S")
echo "exp_date": $exp_date
output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date
echo "output_path": $output_path

python eval.py \
    --model_type  ${model_type} \
    --model_path ${model_path} \
    ${lora_model:+--lora_model "$lora_model"} \
    --cot True \
    --few_shot True \
    --with_prompt False \
    --ntrain 5 \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --do_save_csv True \
    --do_test False \
    --gpus 0,1,2,3 \
    --only_cpu False \
    --output_dir ${output_path}
```
