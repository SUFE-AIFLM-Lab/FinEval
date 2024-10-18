# Prompt_Template

The `with_prompt` in the configuration file can determine whether we use the answer template. Our template is defined as follows.

The Alpaca model is recommended. Other models except the Alpaca model are not recommended to use this template.

It is measured that other models using this template will reduce their scores.

```text
if with_prompt:
                prompt_template = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Response: ")

                instruction = prompt_template.format_map({'instruction': instruction,'subject':subject_name})
```
