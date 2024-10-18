import pandas as pd
import random

# 读取原始 CSV 文件
df = pd.read_csv('tax_law_test.csv')

# 随机抽取 27 道题目
selected_questions = df.sample(n=27, random_state=42)  # 设置 random_state 以确保可重复的结果

# 将抽取的题目保存为新的 CSV 文件
selected_questions.to_csv('tax_law_test_selected.csv', index=False, encoding='utf-8')

print("27 道题目已成功保存为新的 CSV 文件。")