# Prompt overview

In LLMs' Supervised Fine-Tuning (SFT) process, we often inject some predefined strings into the dialogue according to actual requirements, so that the model can output content according to certain requirements.

For example, in the fine-tuning of some `chat` models, we may add system-level instructions at the beginning of each dialog, and agree on a set of formats to represent the dialog between the user and the model. During evaluation, we also need to input questions in the agreed format so that the model can exert its maximum performance.

The format of the prompt is very important! ! Pay attention to line breaks! ! Be careful not to have spaces at the end of each line.

According to the explanation in the parameter configuration, the combination of few-shot and CoT parameters can generate four evaluation methods. According to different evaluation methods, we have different prompts.
