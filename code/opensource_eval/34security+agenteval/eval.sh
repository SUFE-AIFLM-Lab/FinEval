
xuanyuan3_70b=llama3-xuanyuan3-70b-chat

python eval_security.py --model xuanyuan3-70b-chat  --model_name_or_path $xuanyuan3_70b --gpus 2,3 --eval_data all 
python eval_security.py --model xuanyuan3-70b-chat  --model_name_or_path $xuanyuan3_70b --gpus 2,3 --eval_data all --cot

python eval_agent.py --model xuanyuan3-70b-chat  --model_name_or_path $xuanyuan3_70b --gpus 2,3 --eval_data all --cot
