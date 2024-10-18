
qwen2_7b=qwen2-7b

python eval.py --model qwen2-7b --model_name_or_path ${qwen2_7b} --gpus 0,1 --eval_data all
python eval.py --model qwen2-7b --model_name_or_path ${qwen2_7b} --gpus 0,1 --eval_data all --cot




