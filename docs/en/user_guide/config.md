# Parameter Configuration Description

## 1. Configuration instructions

The parameter combination of few-shot and cot can produce four evaluation methods:

- few-shot=False and cot=False: means that zero-shot only answers the answer.
- few-shot=True and cot=False: means that few-shot only answers the answer.
- few-shot=False and cot=True: The zero-shot method adopts the CoT method to answer.
- few-shot=True and cot=True: The few-shot method uses the CoT method to answer.

Generally speaking, the effect of the few-shot Base model in the pretraining stage will be better than zero-shot, but the Chat model that has been aligned with human preferences is likely to have a better zero-shot effect than the Base model.

Different model_types represent different model model reading configurations, and model_type can be selected from the following configurations:

```text
"bloom": (BloomForCausalLM, BloomTokenizerFast),
"chatglm": (AutoModel, AutoTokenizer),
"llama": (LlamaForCausalLM, LlamaTokenizer),
"baichuan": (AutoModelForCausalLM, AutoTokenizer),
"auto": (AutoModelForCausalLM, AutoTokenizer),
"moss":(AutoConfig, AutoTokenizer)
```

## 2. Model configuration information

The following is the model configuration information:

```text
('--model_type', default=None, type=str, required=True): model name, this parameter is required.
("--model_path", type=str): model path, please fill in the correct model path
('--lora_model', default="", type=str, help="If None, perform inference on the base model")
("--cot",choices=["False","True"], default="False"): Whether to use Chain-of-thought (chain of thought), if you use the chain of thought method, set this parameter to True, The default is False, that is, the evaluation is not carried out in the way of thinking chain.
("--few_shot", choices=["False","True"], default="True"): Whether to use the few-shot method for reasoning. If this parameter is used, a small number of examples will be provided to the model for the model to learn .
("--ntrain", "-k", type=int, default=5): the number of few-shot, the default is 5-shot, when few-shot is False, this parameter is invalid
("--with_prompt", choices=["False","True"], default="False"): Whether to use the prompt template of alpaca, the default is not applicable, except for the Alpaca class model, it is not recommended to set it to True for other models.
("--constrained_decoding", choices=["False","True"], default="True"): Whether to use a restricted decoding method. Since the standard answer of fineval is ABCD, two methods are provided to extract from the model Answer scheme: when constrained_decoding=True, calculate the probability that the first token generated by the model is ABCD, and select the one with the highest probability as the answer; when constrained_decoding=False, use regular expressions to extract the answer from the content generated by the model.
("--temperature", type=float, default=0.2): The output temperature is a parameter used to adjust the diversity of the generated results. A lower output temperature value will make the generated results more conservative and deterministic, and more inclined to select words with higher probability in the probability distribution as output. In other words, a lower output temperature results in a more focused and defined result.
("--n_times", default=1, type=int): Specify the number of repetitions of the evaluation, put the model under output_dir to generate a folder with the specified number of times, the default is 1, and the generated folder is toke0
("--do_save_csv", choices=["False","True"], default="False"): Whether to save the generated results of each subject model, standard answers, etc. in a csv file.
("--output_dir", type=str): Specify the output path of the evaluation results and csv files
("--do_test", choices=["False","True"], default="False"): Test on the valid and test sets, when do_test=False, test on the valid set; when do_test=True , to test on the test set, and the default is to test on the valid set.
('--gpus', default="0", type=str): The number of gpus used in model testing, here fill in the gpu number instead of the number. If there are multiple gpus, fill in 0, 1, 2....
('--only_cpu', choices=["False","True"], default="False", help='only use CPU for inference'): whether to use only cpu for inference, the default does not apply cpu for inference
```

