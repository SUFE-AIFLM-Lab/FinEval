# Prompt_Template

配置文件中`with_prompt`，可以决定我们使用否采用回答模板，我们的模板定义如下，Alpaca模型推荐使用，除Alpaca模型外其他模型不推荐使用此模板，经实测其他模型使用该模板分数会降低。

```text
if with_prompt:
                prompt_template = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Response: ")

                instruction = prompt_template.format_map({'instruction': instruction,'subject':subject_name})
```
