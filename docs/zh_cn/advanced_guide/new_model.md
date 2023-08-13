# 支持新模型（选用）

- 如果模型采用AutoModelForCausalLM,AutoTokenizer方式加载,指定model_type（模型名称）为auto，其余参数正常填写，即可加载新模型。

- 如果模型采用其他方式加载(AutoModelForCausalLM,AutoTokenizer无法加载模型)，可修改`/code/evaluators/unify_evaluator.py`文件


1. 自定义增加模型加载信息,修改`/code/evaluators/unify_evaluator.py`文件，在transformers处进行导入此参数：

    ```
    from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoModelForCausalLM,
        BloomForCausalLM,
        BloomTokenizerFast,
        LlamaTokenizer,
        LlamaForCausalLM,
        AutoConfig,
        模型新的加载方式
    )
    ```

2. 加入自定义模型修改信息		

    ```
    MODEL_CLASSES = {
        "bloom": (BloomForCausalLM, BloomTokenizerFast),
        "chatglm": (AutoModel, AutoTokenizer),
        "llama": (LlamaForCausalLM, LlamaTokenizer),
        "baichuan": (AutoModelForCausalLM, AutoTokenizer),
        "auto": (AutoModelForCausalLM, AutoTokenizer),
        "moss":(AutoConfig, AutoTokenizer),
        "自定义模型":(模型加载方式,分词器加载方式)
    }
    ```

3. 在`/code/evaluators/unify_evaluator.py`中加入您新的模型加载逻辑。
