import os
import subprocess
import datetime

# Set environment variables
os.environ['PROJ_HOME'] = os.getcwd()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Input your OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
base_url = 'https://www.apillm.online/v1'
exp_name = 'claude-3-5-sonnet-20240620'
exp_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
output_path = os.path.join(os.environ['PROJ_HOME'], 'output_dir', exp_name, exp_date)

print(f"exp_date: {exp_date}")
print(f"output_path: {output_path}")

# Build the command to run the Python script
command = [
    'python', 'eval_chatgpt.py',
    '--api_key', api_key,
    '--base_url', base_url,
    '--cot', 'True',
    '--few_shot', 'True',
    '--n_times', '1',
    '--ntrain', '5',
    '--do_test', 'True',
    '--do_save_csv', 'True',
    '--output_dir', output_path,
    '--model_name', exp_name
]

# Execute the command
subprocess.run(command, check=True)
