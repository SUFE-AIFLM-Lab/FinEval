import os
from tqdm import tqdm
import jieba
from rouge_chinese import Rouge
from utils import _remove_punctuation, _mixed_segmentation, _find_lcs, write_json, load_json, \
    extract_questions_and_text, _compute_f1_score, compute, extract_cotanswer, extract_cotsuggestion
from FlagEmbedding import FlagModel
import numpy
import re
DATA_PATH = 'data'
INSTRUCT_SAMPLES = load_json('data/instruct_samples.json')
COT_INSTRUCT_SAMPLES = load_json('data/instruct_samples_cot.json')


#subjective
# 金融文本分类
class FinNLEvaluator:
    dataset = 'finnl'

    zero_shot_prompts = [
        '请根据新浪财经的新闻，分析出与上下文内容描述相关的金融新闻类别，并只在给出的选择范围中进行选择2-3个，并将多个类别用逗号进行分割。请遵循以下指示：直接输出类别，以最精炼的形式，不需要输出原因及多余文字。\n\可选择的范围：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n\n上下文：{inp}\n答案：',
        '下面给出了一则财经新闻，你可以帮我判断它属于哪一个类别吗？你可以选择的类别包括：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n\n请注意，你只可以选择上述类别中的2-3项，以逗号作分隔，并且你只需要输出类别名称，不要输出其他内容。\n文本：{inp}\n答案：',
        #'参考下面给出的新浪财经的新闻，以及给出的可选择的新闻类别，选出2-3个与内容描述相关的金融新闻类别。\n请注意，你只可以选择上述类别中的2-3项，以逗号作分隔，并且你只需要输出类别名称，不要输出其他内容。\n\n上下文：{inp}\n可选择的范围：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n答案：',
        #'### 任务描述：<br> 你将阅读一段金融新闻文本。请判断该事件属于以下类别中的哪一个。<br> ### 回答样式 ：<br>你只需要从上面的类别中选择一项最贴切的一项，并只需要输出类别名称。注意你只能选择1个类别作为输出。注意不需要输出除了类别外的任何信息，包括你给出判断的原因和解释等都不需要输出。事实上你应当只回答一个词语，而不是完整的句子。<br> 可选的类别有：公司、行业、大盘、国际、经济、政策、政治、期货、债券、房地产、外汇、虚拟货币、其它。<br>  ### 金融新闻文本<br> {inp}\n答案：'
    ]

    zero_shot_cot = [
        '请根据新浪财经的新闻，分析出与上下文内容描述相关的金融新闻类别，并只在给出的选择范围中进行选择2-3个，并将多个类别用逗号进行分割。可选择的范围：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n上下文：{inp}\n',
    ]

    few_shot_prompts = [
        '请根据新浪财经的新闻，分析出与上下文内容描述相关的金融新闻类别，并只在给出的选择范围中进行选择2-3个，并将多个类别用逗号进行分割。请遵循以下指示：直接输出类别，以最精炼的形式，不需要输出原因及多余文字。下面给出了一个样例，按照此样例输出最后一个的答案。\n可选择的范围：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n{context}',
        '下面给出了一些财经新闻，你可以帮我判断它属于哪一个类别吗？你可以选择的类别包括：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n\n请注意，你只可以选择上述类别中的2-3项，以逗号作分隔，并且你只需要输出类别名称，不要输出其他内容。同时，你可以参考下面的样例，然后你需要给出最后一则文本的分类。\n{context}',
        #'参考下面给出的新浪财经的新闻，以及给出的可选择的新闻类别：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n选出2-3个与内容描述相关的金融新闻类别。\n请注意，你只可以选择上述类别中的2-3项，以逗号作分隔，并且你只需要输出类别名称，不要输出其他内容。\n{context}'
    ]

    few_shot_cot = [
        '请根据新浪财经的新闻，分析出与上下文内容描述相关的金融新闻类别，并只在给出的选择范围中进行选择2-3个，并将多个类别用逗号进行分割。可选择的范围：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self,cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)
        self.cot = cot
        self.cot_instructs = COT_INSTRUCT_SAMPLES.get(self.dataset)


    @staticmethod
    def build_zero_shot_prompt(prompt, inp):
        return prompt.format(inp=inp)

    def build_few_shot_prompt(self, prompt, inp1: str, k: int):
        # 基于给定的例子，构建few shot模板
        if self.cot:
            self.instructs = self.cot_instructs
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{inp}\n答案：{answer}'.format(
                inp=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n答案：'.format(inp=inp1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    def evaluate(self, golds, preds, sep):
        if self.cot: preds = [extract_cotanswer(i) for i in preds]
        assert len(golds) == len(preds)
        n1, n2, n3 = 0, 0, 0
        for reference, prediction in zip(golds, preds):
            reference = reference.split(' ')
            #_prediction = prediction.split(sep)
            prediction.replace('\n', '')
            _prediction = re.split(r'[，,、]', prediction)
            # n1表示预测对的实体总个数
            n1 += len(set(reference).intersection(set(_prediction)))
            # n2表示所有的实体数
            n2 += len(reference)
            # n3表示预测的实体数
            n3 += len(_prediction)
        p = n1 / n3
        r = n1 / n2
        if p + r != 0:
            f1 = 2 * ((p * r) / (p + r)) * 100
        else:
            f1 = 0
        return round(f1, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp = example['input']
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp = example['input']
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        if self.cot:
            self.zero_shot_prompts = self.zero_shot_cot
            self.few_shot_prompts = self.few_shot_cot

        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                inp = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds, sep=','))
            all_zero_shot += self.evaluate(golds, preds, sep=',')
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                inp = example['input']
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, k=few_shot_k)
                inputs.append(input_text)
                print(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds, sep=','))
            all_few_shot += self.evaluate(golds, preds, sep=',')
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero, model_response_few, {'zero_shot_metrics_finnl': zero_shot_metrics, 'few_shot_metrics_finnl': few_shot_metrics,
                'avg_zero_shot_finnl': avg_zero_shot, 'avg_few_shot_finnl': avg_few_shot}

# 金融情感分析
class FinFETextEvaluator:
    dataset = 'finfetext'

    zero_shot_prompts = [
        '请根据上下文，分析金融语句的情绪，选项为积极、消极、中性，请在这三个选项中选出唯一正确的选项。请仅输出情绪类别，多余文字不要输出。\n\n上下文：{context}\n选项：积极、消极、中性\n答案：',
        '下面是一段金融语句，你可以告诉我该语句的情绪倾向是什么吗？积极、消极还是中性？你只需要回答情绪“积极”、“消极”或“中性”，不需要给出其他内容。\n\n上下文：{context}\n答案：',
        #'上下文：{context}\n请根据上下文，选出此文本所表现出的情绪，选项为积极、消极、中性。只需要回答情绪“积极”、“消极”或“中性”，不需要给出其他内容。\n答案：',
    ]
    zero_shot_cot = [
        '请根据上下文，分析金融语句的情绪，选项为积极、消极、中性，请在这三个选项中选出唯一正确的选项。\n\n选项：积极、消极、中性\n\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n上下文：{context}\n',
    ]

    few_shot_prompts = [
        '请根据上下文，分析金融语句的情绪，选项为积极、消极、中性，请在这三个选项中选出唯一正确的选项。请遵循以下示例，仅输出情绪类别，多余文字不要输出。下面给出了一些样例，按照样例输出答案。\n{context}',
        '下面是一段金融语句，你可以告诉我该语句的情绪倾向是什么吗？积极、消极还是中性？请参考下面的例子进行回答。\n{context}',
        #'请根据上下文，选出此文本所表现出的情绪，选项为积极、消极、中性。只需要回答情绪“积极”、“消极”或“中性”，不需要给出其他内容。请参考下面的例子进行回答。\n{context}'
    ]
    few_shot_cot = [
        '请根据上下文，分析金融语句的情绪，选项为积极、消极、中性，请在这三个选项中选出唯一正确的选项。\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n下面给出了一些样例，按照样例输出答案。\n{context}',    
    ]

    def __init__(self, cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)
        self.cot = cot
        self.cot_instructs = COT_INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    def build_few_shot_prompt(self, prompt, context: str, k: int):
        # 基于给定的例子，构建few shot模板
        if self.cot:
            self.instructs = self.cot_instructs
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{context}\n选项：积极、消极、中性\n答案：{answer}'.format(
                context=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{context}\n选项：积极、消极、中性\n答案：'.format(context=context)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    def evaluate(self, golds, preds):
        if self.cot: preds = [extract_cotanswer(i) for i in preds]

        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            # 如果需要处理文本，可以调用 clean 函数，例如 if gold == clean(pred):
            if gold == _remove_punctuation(pred):
                s += 1
        return round(s / len(golds) * 100, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], context=example['input'], k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        if self.cot:
            self.zero_shot_prompts = self.zero_shot_cot
            self.few_shot_prompts = self.few_shot_cot

        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                inputs.append(input_text)                
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, context=example['input'], k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero, model_response_few, {'zero_shot_metrics_finfetext': zero_shot_metrics, 'few_shot_metrics_finfetext': few_shot_metrics,
                'avg_zero_shot': avg_zero_shot, 'avg_few_shot': avg_few_shot}

# 关联关系抽取
class FinREEvaluator:
    dataset = 'finre'

    zero_shot_prompts = [
        '### 任务描述：<br> 你将阅读一段金融事件。请判断该事件属于以下类别中的哪一个。<br> 可选的类别有：增持、收购、发行、订单、重组、转让、注资、借壳、成立、合并、减持、入股、合资、商讨、签约、上市、合作、中标、其他。<br> ### 回答规范 ：<br>你只需要从上面的类别中选择一项最贴切的即可。注意你只能选择1个类别作为输出。注意不需要输出其他任何信息，包括你给出判断的原因和解释等都不需要输出。<br>  ### 金融事件<br> {context}\n答案：',
    ]
    zero_shot_cot = [
        '### 任务描述：<br> 你将阅读一段金融事件。请判断该事件属于以下类别中的哪一个。<br> 可选的类别有：增持、收购、发行、订单、重组、转让、注资、借壳、成立、合并、减持、入股、合资、商讨、签约、上市、合作、中标、其他。<br> \
        ### 回答规范 ：<br>你只需要从上面的类别中选择一项最贴切的即可。注意你只能选择1个类别作为输出。注意不需要输出其他任何信息，包括你给出判断的原因和解释等都不需要输出。<br>  \
        # 一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n### 金融事件<br> {context}\n答案：',
    ]

    few_shot_prompts = [
        '请根据财经金融领域文本及问题，在给定的关系选项里进行问题答案的选择，仅选择唯一正确的一个答案，并直接输出答案，不要输出任何其他文字内容。\n\n可能的关系选项为：增持、收购、发行、订单、重组、转让、注资、借壳、成立、合并、减持、入股、合资、商讨、签约、上市、合作、中标、其他\n请在上面这些类别里对两个主体的关系进行选择。如果没有特殊的关系，请输出其他。下面给出示例：\n\n{context}',
        '请根据财经文本的文本，回答问题。你答案可以选择的关系类别包括： 增持、收购、发行、订单、重组、转让、注资、借壳、成立、合并、减持、入股、合资、商讨、签约、上市、合作、中标、其他\n请注意，你只可以选择一个类别，如果两者之间的关系不属于上面这些类别，你可以请输出其他。下面给出示例：\n\n{context}',
        '参考材料，请问给出的两个主体的关系属于下面给出类别里的哪个？你答案可以选择的关系类别包括：增持、收购、发行、订单、重组、转让、注资、借壳、成立、合并、减持、入股、合资、商讨、签约、上市、合作、中标、其他\n\n下面给出示例：\n\n{context}'
    ]
    few_shot_cot = [
        '请根据财经金融领域文本及问题，在给定的关系选项里进行问题答案的选择，仅选择唯一正确的一个答案，并直接输出答案，不要输出任何其他文字内容。\n\n可能的关系选项为：增持、收购、发行、订单、重组、转让、注资、借壳、成立、合并、减持、入股、合资、商讨、签约、上市、合作、中标、其他\n\
        请在上面这些类别里对两个主体的关系进行选择。如果没有特殊的关系，请输出其他。一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n下面给出示例：\n\n{context}',
    ]

    def __init__(self, cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)
        self.cot = cot
        self.cot_instructs = COT_INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    def build_few_shot_prompt(self, prompt, inp1: str, k: int):
        # 基于给定的例子，构建few shot模板
        if self.cot:
            self.instructs = self.cot_instructs
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{inp}\n请在上面这些类别里对两个主体的关系进行选择。\n答案：{answer}'.format(
                inp=instruct['input'],
                answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n请对这两个主体的关系进行选择。如果没有特殊的关系，请输出其他。\n答案：'.format(inp=inp1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    def evaluate(self, golds, preds):
        if self.cot: preds = [extract_cotanswer(i) for i in preds]
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            terms = "增持|收购|发行|订单|重组|转让|注资|借壳|成立|合并|减持|入股|合资|商讨|签约|上市|合作|中标|其他"
            match = re.search(terms, pred)#匹配第一个匹配项
            ans = match.group(0) if match else None
            if gold == ans:
                s += 1
        return round(s / len(golds) * 100, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        context = example['input']
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=context))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp = example['input']
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, k=1))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        if self.cot:
            self.zero_shot_prompts = self.zero_shot_cot
            self.few_shot_prompts = self.few_shot_cot

        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                context = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=context)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                inp = example['input']
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp,
                                                        k=few_shot_k)
                #print(input_text)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero, model_response_few, {'zero_shot_metrics_finre': zero_shot_metrics, 'few_shot_metrics_finre': few_shot_metrics,
                'avg_zero_shot_finre': avg_zero_shot, 'avg_few_shot_finre': avg_few_shot}

# 金融事件抽取
class FinQAEvaluator:
    dataset = 'finqa'

    zero_shot_prompts = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。\n\n上下文: {inp}\n问题: {ins}\n答案：',
        '我需要从下面的文本中识别事件信息，你可以帮助我回答下面的阅读理解问题吗？\n\n上下文：{inp}\n问题：{ins}\n答案：',
        '上下文：{inp}\n问题：{ins}请根据此上下文及问题，回答答案。'
    ]
    zero_shot_cot = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。\n\n\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n上下文: {inp}\n问题: {ins}\n',
    ]

    few_shot_prompts = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
        '我需要从下面的文本中识别事件信息，你可以帮助我回答下面的阅读理解问题吗？你的回答可以参考下面的样例。\n\n{context}',
        '请根据提供的文本信息，回答问题。下面给出了一个例子：\n{context}'
    ]
    few_shot_cot = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]



    def __init__(self, cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)
        self.cot = cot
        self.cot_instructs = COT_INSTRUCT_SAMPLES.get(self.dataset)


    @staticmethod
    def build_zero_shot_prompt(prompt, inp, ins):
        return prompt.format(inp=inp, ins=ins)

    def build_few_shot_prompt(self, prompt, inp1: str, ins1: str, k: int):
        # 基于给定的例子，构建few shot模板
        if self.cot:
            self.instructs = self.cot_instructs
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            ins, inp = extract_questions_and_text(instruct['input'])
            instruct_prompts.append('上下文：{inp}\n问题：{ins}\n答案：{answer}'.format(
                ins=ins, inp=inp, answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n问题：{ins}\n答案：'.format(inp=inp1, ins=ins1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    def evaluate(self, golds, preds):
        if self.cot: preds = [extract_cotanswer(i) for i in preds]
        assert len(golds) == len(preds)
        f1, total_count = 0, 0
        for gold, pred in zip(golds, preds):
            pred = _mixed_segmentation(pred, rm_punc=True)
            gold = _mixed_segmentation(gold, rm_punc=True)
            lcs, lcs_len = _find_lcs(gold, pred)
            if lcs_len == 0:
                score = 0
            else:
                precision = 1.0 * lcs_len / len(pred)
                recall = 1.0 * lcs_len / len(gold)
                score = (2 * precision * recall) / (precision + recall)
            total_count += 1
            f1 += score
        f1_score = 100.0 * f1 / total_count

        return round(f1_score, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        ins, inp = extract_questions_and_text(example['input'])
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp, ins=ins))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        ins, inp = extract_questions_and_text(example['input'])
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, ins1=ins, k=1))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        if self.cot:
            self.zero_shot_prompts = self.zero_shot_cot
            self.few_shot_prompts = self.few_shot_cot

        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                ins, inp = extract_questions_and_text(example['input'])
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp, ins=ins)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.few_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                ins, inp = extract_questions_and_text(example['input'])
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, ins1=ins, k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero, model_response_few, {'zero_shot_metrics_finqa': zero_shot_metrics, 'few_shot_metrics_finqa': few_shot_metrics,
                'avg_zero_shot_finqa': avg_zero_shot, 'avg_few_shot_finqa': avg_few_shot}

# 负面消息确定：
class FinNSP1Evaluator:

    dataset = 'finnsp1'

    zero_shot_prompts = [
        '请根据上下文，判定该文本是否包含金融实体的负面信息。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。如果包含金融实体的负面信息，则输出1。请仅输出0或1。\n\n上下文：{inp}\n答案：',
        '文本：{inp}\n上面的文本中是否包含某个金融实体的负面信息呢？如果没有上文没有包含某一个实体的负面信息，或是包含负面信息但是没有提到负面信息相关的实体，请回答0；如果既包含负面信息，又提到了涉及负面信息的金融实体，请回答1。请注意，你的回答只能为0或1。',
        '请根据下面给出的材料，判断出该文本是否包含金融实体的负面信息。如果包含金融实体的负面信息，则输出1。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。请仅输出0或1。\n\n上下文：{inp}\n答案：'
    ]
    zero_shot_cot = [
        '请根据上下文，判定该文本是否包含金融实体的负面信息。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。如果包含金融实体的负面信息，则输出1。请仅输出0或1。\n\n\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n上下文：{inp}\n',
    ]

    few_shot_prompts = [
        '请根据上下文，判定该文本是否包含金融实体的负面信息。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。如果包含金融实体的负面信息，则输出1。请遵循以下示例，仅输出0或1。下面给出了几个样例，按照此样例输出最后一个的答案。\n{context}',
        '下面的文本中是否包含某个金融实体的负面信息呢？如果没有上文没有包含某一个实体的负面信息，或是包含负面信息但是没有提到负面信息相关的实体，请回答0；如果既包含负面信息，又提到了涉及负面信息的金融实体，请回答1。你可以参考下面的几个例子，并给出最后一个例子的答案。\n{context}',
        '请根据下面给出的材料，判断出该文本是否包含金融实体的负面信息。如果包含金融实体的负面信息，则输出1。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。请仅输出0或1。\n\n下面给出几个示例：\n{context}'
    ]
    few_shot_cot = [
        '请根据上下文，判定该文本是否包含金融实体的负面信息。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。如果包含金融实体的负面信息，则输出1。请遵循以下示例，仅输出0或1。\n\n\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n下面给出了几个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self, cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)
        self.cot = cot
        self.cot_instructs = COT_INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp):
        return prompt.format(inp=inp)

    def build_few_shot_prompt(self, prompt, inp1: str, k: int):
        # 基于给定的例子，构建few shot模板
        if self.cot:
            self.instructs = self.cot_instructs
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{inp}\n答案：{answer}'.format(
                inp=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n答案：'.format(inp=inp1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    def evaluate(self, golds, preds):
        if self.cot: preds = [extract_cotanswer(i) for i in preds]
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            # 如果需要处理文本，可以调用 clean 函数，例如 if gold == clean(pred):
            if gold == _remove_punctuation(pred):
                s += 1
        return round(s / len(golds) *100, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp = example['input']
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp = example['input']
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, k=2))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        if self.cot:
            self.zero_shot_prompts = self.zero_shot_cot
            self.few_shot_prompts = self.few_shot_cot

        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                inp = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                inp = example['input']
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero, model_response_few, {'zero_shot_metrics_finnsp1': zero_shot_metrics, 'few_shot_metrics_finnsp1': few_shot_metrics, 'avg_zero_shot_finnsp1' : avg_zero_shot, 'avg_few_shot_finnsp1' : avg_few_shot}

# 负面实体抽取
class FinNSP2Evaluator:
    dataset = 'finnsp2'

    zero_shot_prompts = [
        '根据上下文，判断负面信息的主体对象是给定的实体列表中的哪些实体，输出这些选出的实体。如果是多个实体，用逗号进行分割。请直接输出实体，不要输出任何其他文字，并用逗号把多个实体进行分割。\n\n上下文：{inp}\n答案：',
        '下面的文本中包含了一些负面信息，同时还给出了一个实体列表，你能帮我抽取到负面信息的主体对象是什么吗？请注意，你的主体对象必须在实体列表中。你的回答只需要包含实体名称，如果要输出多个实体，请用逗号作分隔。\n\n上下文：{inp}\n答案：',
        '请参考给出的材料信息，在给定的实体列表中，选出并输出负面信息的主体对象是哪些实体。如果是多个实体，用逗号进行分割。请直接输出实体，不要输出任何其他文字，并用逗号把多个实体进行分割。\n\n材料：{inp}\n答案：',
        #"### 任务描述：<br> 请首先从以下这段金融新闻中抽取事件主体，然后判断这份金融新闻是否含有负面消息，如果含有负面消息且与事件主体有关输出1，如果含有负面消息但与主体无关或不含有负面消息，则输出0，最终以[主体，0或1]格式输出，直接输出无需分析<br> ### 回答规范 ：<br>你只需要将[主体，0或1] 中的主体替换为具体内容，0/1中做出选择，将答案填写在[ ] 中。注意不需要输出其他任何信息，包括你给出判断的原因和解释等都不需要输出。只需要给出最终方括号和其中的判断即可。一个可行的输出形如'[纺织服装业,0]' <br>  ### 金融新闻<br> \n{inp}\n答案："
    ]

    few_shot_prompts = [
        '根据上下文，判断负面信息的主体对象是给定的实体列表中的哪些实体，输出这些选出的实体。如果是多个实体，用逗号进行分割。请直接输出实体，不要输出任何其他文字，并用逗号把多个实体进行分割。下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
        '下面的文本中包含了一些负面信息，同时还给出了一个实体列表，你能帮我抽取到负面信息的主体对象是什么吗？请注意，你的主体对象必须在实体列表中。你的回答只需要包含实体名称，如果要输出多个实体，请用逗号作分隔。\n你可以参考下面的样例，然后你需要给出最后一个问题的答案。\n{context}',
        '请参考给出的材料信息，在给定的实体列表中，选出并输出负面信息的主体对象是哪些实体。如果是多个实体，用逗号进行分割。请直接输出实体，不要输出任何其他文字，并用逗号把多个实体进行分割。\n{context}'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp):
        return prompt.format(inp=inp)

    def build_few_shot_prompt(self, prompt, inp1: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{inp}\n答案：{answer}'.format(
                inp=instruct['input'][0], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n答案：'.format(inp=inp1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds, sep):
        assert len(golds) == len(preds)
        n1, n2, n3 = 0, 0, 0
        for reference, prediction in zip(golds, preds):
            reference = reference.split(';')
            _prediction = prediction.split(sep)
            # n1表示预测对的实体总个数
            n1 += len(set(reference).intersection(set(_prediction)))
            # n2表示所有的实体数
            n2 += len(reference)
            # n3表示预测的实体数
            n3 += len(_prediction)
        p = n1 / n3
        r = n1 / n2
        if p + r != 0:
            f1 = 2 * ((p * r) / (p + r)) * 100
        else:
            f1 = 0
        return round(f1, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp, ins = example['input'][0], example['input'][1]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp, ins = example['input'][0], example['input'][1]
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                inp = example['input'][0]
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds, sep='，'))
            all_zero_shot += self.evaluate(golds, preds, sep='，')
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                inp = example['input'][0]
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds, sep='，'))
            all_few_shot += self.evaluate(golds, preds, sep='，')
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero, model_response_few, {'zero_shot_metrics_finnsp2': zero_shot_metrics, 'few_shot_metrics_finnsp2': few_shot_metrics,
                'avg_zero_shot_finnsp2': avg_zero_shot, 'avg_few_shot_finnsp2': avg_few_shot}

# 因果事件抽取
class FinCQAEvaluator:

    dataset = 'fincqa'

    zero_shot_prompts = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。你只需要输出答案，不要输出其他内容。\n\n上下文: {inp}\n问题: {ins}\n答案：',
        '我需要从下面的文本中识别事件信息，你可以帮助我回答下面的阅读理解问题吗？你只需要输出答案，不要输出其他内容。\n\n上下文：{inp}\n问题：{ins}\n答案：',
        '上下文：{inp}\n问题：{ins}请根据此上下文及问题，回答答案。你只需要输出答案，不要输出其他内容。'
    ]
    zero_shot_cot = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。\n\n\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n上下文: {inp}\n问题: {ins}\n',
    ]

    few_shot_prompts = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。下面给出了一个样例，按照此样例输出最后一个的答案。你只需要输出答案，不要输出其他内容。\n{context}',
        '我需要从下面的文本中识别事件信息，你可以帮助我回答下面的阅读理解问题吗？你的回答可以参考下面的样例。你只需要输出答案，不要输出其他内容。\n\n{context}',
        '请根据提供的文本信息，回答问题。你只需要输出答案，不要输出其他内容。下面给出了一个例子：\n{context}'
    ]
    few_shot_cot = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。\
        一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你的最终答案}}\n下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self, cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)
        self.cot = cot
        self.cot_instructs = COT_INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp, ins):
        return prompt.format(inp=inp, ins=ins)

    def build_few_shot_prompt(self, prompt, inp1: str, ins1: str, k: int):
        # 基于给定的例子，构建few shot模板
        if self.cot:
            self.instructs = self.cot_instructs

        instruct_prompts = []
        for instruct in self.instructs[: k]:
            ins, inp = extract_questions_and_text(instruct['input'])
            instruct_prompts.append('上下文：{inp}\n问题：{ins}\n答案：{answer}'.format(
                ins=ins, inp=inp, answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n问题：{ins}\n答案：'.format(inp=inp1, ins=ins1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)


    def evaluate(self, golds, preds):
        if self.cot: preds = [extract_cotanswer(i) for i in preds]
        assert len(golds) == len(preds)
        f1, total_count = 0, 0
        for gold, pred in zip(golds, preds):
            pred = _mixed_segmentation(pred, rm_punc=True)
            gold = _mixed_segmentation(gold, rm_punc=True)
            lcs, lcs_len = _find_lcs(gold, pred)
            if lcs_len == 0:
                score = 0
            else:
                precision = 1.0 * lcs_len / len(pred)
                recall = 1.0 * lcs_len / len(gold)
                score = (2 * precision * recall) / (precision + recall)
            total_count += 1
            f1 += score
        f1_score = 100.0 * f1 / total_count

        return round(f1_score, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        ins, inp = extract_questions_and_text(example['input'])
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp, ins=ins))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        ins, inp = extract_questions_and_text(example['input'])
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, ins1=ins, k=1))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        if self.cot:
            self.zero_shot_prompts = self.zero_shot_cot
            self.few_shot_prompts = self.few_shot_cot

        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                ins, inp = extract_questions_and_text(example['input'])
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp, ins=ins)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                ins, inp = extract_questions_and_text(example['input'])
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, ins1=ins, k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot  += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero, model_response_few, {'zero_shot_metrics_fincqa' : zero_shot_metrics, 'few_shot_metrics_fincqa' : few_shot_metrics, 'avg_zero_shot_fincqa' : avg_zero_shot, 'avg_few_shot_fincqa' : avg_few_shot}

#objective
# 金融摘要任务
class FinNAEvaluator:
    dataset = 'finna'

    zero_shot_prompts = [
        '请根据上下文给出的中文短新闻，生成对应的不超过20个字的摘要。\n\n上下文：{context}',
        '新闻：{context}\n你可以帮助我归纳一个不超过20字的摘要吗？',
        #'上下文：{context}\n请根据上下文给出的新闻，生成对应的不超过20个字的简短摘要。'
    ]
    zero_shot_cot = [
        '请根据上下文给出的中文短新闻，生成对应的不超过20个字的摘要。\n\n\
            一步一步思考并在最后遵循以下格式输出：\n答案：{{这是最终的答案}}\n上下文：{context}',
    ]

    few_shot_prompts = [
        '请根据上下文给出的中文短新闻，生成对应的不超过20个字的摘要。\n\n下面给出了一些样例，按照样例输出最后一个的答案。\n{context}',
        '你可以帮助我归纳一个不超过20字的摘要吗？下面给出了一些样例，按照样例输出最后一个的答案。\n{context}',
        #'请根据上下文给出的新闻，生成对应的不超过20个字的简短摘要。下面给出了一些样例，按照样例输出最后一个的答案。\n{context}'
    ]
    few_shot_cot = [
        '请根据上下文给出的中文短新闻，生成对应的不超过20个字的摘要。\n\n\
            一步一步思考并在最后遵循以下格式输出：\n答案：{{这是最终的答案}}\n下面给出了一些样例，按照样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self, cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)
        self.cot = cot
        self.cot_instructs = COT_INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    def build_few_shot_prompt(self, prompt, context: str, k: int):
        # 基于给定的例子，构建few shot模板
        if self.cot:
            self.instructs = self.cot_instructs
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('请根据上下文给出的中文短新闻，生成对应的不超过20个字的摘要。上下文：{context}\n答案：{answer}'.format(
                context=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '请根据上下文给出的中文短新闻，生成对应的不超过20个字的摘要。上下文：{context}\n答案：'.format(context=context)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    def evaluate(self, golds, preds):
        if self.cot: preds = [extract_cotanswer(i) for i in preds]
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            string2 = pred
            string1 = gold
            # 创建 Rouge 对象
            rouge = Rouge()
            # 对字符串进行分词
            string1_tokens = jieba.cut(string1)
            string2_tokens = jieba.cut(string2)
            string1_text = " ".join(string1_tokens)
            string2_text = " ".join(string2_tokens)
            print(string1_text, string2_text)
            scores = rouge.get_scores(string1_text, string2_text)
            rouge_l_f1 = scores[0]['rouge-l']['f']
            s += rouge_l_f1
        num_all = len(golds)
        acc_rate = s / num_all * 100
        return round(acc_rate, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        if self.cot:
            self.zero_shot_prompts = self.zero_shot_cot
            self.few_shot_prompts = self.few_shot_cot

        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, context=example['input'], k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero,model_response_few,{'zero_shot_metrics_finna': zero_shot_metrics, 'few_shot_metrics_finna': few_shot_metrics,
                'avg_zero_shot': avg_zero_shot, 'avg_few_shot': avg_few_shot}

# 金融用户画像
class FinCustomerEvaluator:
    dataset = 'fincustomer'

    zero_shot_prompts = [
        '请根据上下文，分析客户描述进行客户画像，选项为年轻白领、中年企业家、退休老人，金融专业人士、互联网创业者、家庭主妇、学生、自由职业者、领域专家、高净值人士、中小企业主、事业单位员工。请在这12个选项中选出唯一正确的选项。请仅输出选项类别，多余文字不要输出。\n\n客户描述：{context}\n选项：年轻白领、中年企业家、退休老人，金融专业人士、互联网创业者、家庭主妇、学生、自由职业者、领域专家、高净值人士、中小企业主、事业单位员工\n答案：',
    ]
    zero_shot_cot = [
        '请根据上下文，分析客户描述进行客户画像，选项为年轻白领、中年企业家、退休老人，金融专业人士、互联网创业者、家庭主妇、学生、自由职业者、领域专家、高净值人士、中小企业主、事业单位员工。请在这12个选项中选出唯一正确的选项。请仅输出选项类别，多余文字不要输出。\n\n\
            客户描述：{context}\n选项：年轻白领、中年企业家、退休老人，金融专业人士、互联网创业者、家庭主妇、学生、自由职业者、领域专家、高净值人士、中小企业主、事业单位员工\n\
            一步一步思考并在最后遵循以下格式输出：\n答案：{{这是最终的答案}}',
    ]

    few_shot_prompts = [
        '请根据上下文，分析客户描述进行客户画像，选项为年轻白领、中年企业家、退休老人，金融专业人士、互联网创业者、家庭主妇、学生、自由职业者、领域专家、高净值人士、中小企业主、事业单位员工。请在这12个选项中选出唯一正确的选项。请仅输出选项类别，多余文字不要输出。下面给出了一些样例，按照样例输出答案。\n{context}',
    ]
    few_shot_cot = [
        '请根据上下文，分析客户描述进行客户画像，选项为年轻白领、中年企业家、退休老人，金融专业人士、互联网创业者、家庭主妇、学生、自由职业者、领域专家、高净值人士、中小企业主、事业单位员工。请在这12个选项中选出唯一正确的选项。请仅输出选项类别，多余文字不要输出。\n\n\
            一步一步思考并在最后遵循以下格式输出：\n答案：{{这是最终的答案}}\
            下面给出了一些样例，按照样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self, cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)
        self.cot = cot
        self.cot_instructs = COT_INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    def build_few_shot_prompt(self, prompt, context: str, k: int):
        # 基于给定的例子，构建few shot模板
        if self.cot:
            self.instructs = self.cot_instructs
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('客户描述：{context}\n选项：年轻白领、中年企业家、退休老人，金融专业人士、互联网创业者、家庭主妇、学生、自由职业者、领域专家、高净值人士、中小企业主、事业单位员工\n答案：{answer}'.format(
                context=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '客户描述：{context}\n选项：年轻白领、中年企业家、退休老人，金融专业人士、互联网创业者、家庭主妇、学生、自由职业者、领域专家、高净值人士、中小企业主、事业单位员工\n答案：'.format(context=context)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    def evaluate(self, golds, preds):
        if self.cot: preds = [extract_cotanswer(i) for i in preds]
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            # 如果需要处理文本，可以调用 clean 函数，例如 if gold == clean(pred):
            if gold == _remove_punctuation(pred):
                s += 1
        return round(s / len(golds) * 100, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], context=example['input'], k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        if self.cot:
            self.zero_shot_prompts = self.zero_shot_cot
            self.few_shot_prompts = self.few_shot_cot

        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, context=example['input'], k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero,model_response_few,{'zero_shot_metrics_fincustomer': zero_shot_metrics, 'few_shot_metrics_fincustomer': few_shot_metrics,
                'avg_zero_shot': avg_zero_shot, 'avg_few_shot': avg_few_shot}

# 营销数据
class FinSalesEvaluator:
    dataset = 'finsales'

    zero_shot_prompts = [
        '请解释下面金融营销术语:\n{context}'
    ]

    few_shot_prompts = [
        '请解释下面金融营销术语\n下面给出了一些样例，按照样例输出最后一个金融营销术语的答案。\n{context}'
    ]


    def __init__(self, cot=False):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    def build_few_shot_prompt(self, prompt, context: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('请解释下面金融营销术语:{context}\n答案：{answer}'.format(
                context=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '请解释下面金融营销术语:{context}\n答案：'.format(context=context)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        sentences_1 = golds
        sentences_2 = preds
        model = FlagModel('/root/MyData/data/myModel/bge-large-zh',
                          query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                          use_fp16=True)
        embeddings_1 = model.encode(sentences_1)
        embeddings_2 = model.encode(sentences_2)
        similarity = embeddings_1 @ embeddings_2.T
        sum_ = sum(similarity)
        print(sum_)

        return round(numpy.sum(sum_) / len(similarity), 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, context=example['input'], k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero,model_response_few,{'zero_shot_metrics_finsales': zero_shot_metrics, 'few_shot_metrics_finsales': few_shot_metrics,
                'avg_zero_shot': avg_zero_shot, 'avg_few_shot': avg_few_shot}

# 金融投资建议
class FinSuggestionEvaluator:
    dataset = 'finsuggestion'

    zero_shot_prompts = [
        '你是一个提供金融建议的专家。{context}'
    ]
    zero_shot_cot = [
        '你是一个提供金融建议的专家。一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你提供的金融建议}}\n{context}'
    ]

    few_shot_prompts = [
        '你是一个提供金融建议的专家。下面给出了一些样例，按照样例输出最后一个金融建议。{context}'
    ]
    few_shot_cot = [
        '你是一个提供金融建议的专家。一步一步思考并在最后遵循以下格式输出：\n答案：{{这是你提供的金融建议}}\n\
        下面给出了一些样例，按照样例输出最后一个金融建议。{context}'
    ]

    def __init__(self, cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)
        self.cot = cot
        self.cot_instructs = COT_INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    
    def build_few_shot_prompt(self, prompt, context: str, k: int):
        # 基于给定的例子，构建few shot模板
        if self.cot:
            self.instructs = self.cot_instructs
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('{context}\n建议：{answer}'.format(
                context=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '{context}\n建议：'.format(context=context)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    def evaluate(self, golds, preds):
        if self.cot: preds = [extract_cotsuggestion(i) for i in preds]
        assert len(golds) == len(preds)
        sentences_1 = golds
        sentences_2 = preds
        model = FlagModel('/root/MyData/data/myModel/bge-large-zh',
                          query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                          use_fp16=True)
        embeddings_1 = model.encode(sentences_1)
        embeddings_2 = model.encode(sentences_2)
        similarity = embeddings_1 @ embeddings_2.T
        sum_ = sum(similarity)
        print(sum_)

        return round(numpy.sum(sum_) / len(similarity), 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))


    def run_evaluation(self, llm, few_shot_k: int = 3):
        if self.cot:
            self.zero_shot_prompts = self.zero_shot_cot
            self.few_shot_prompts = self.few_shot_cot

        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, context=example['input'], k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero,model_response_few,{'zero_shot_metrics_finsuggestion': zero_shot_metrics, 'few_shot_metrics_finsuggestion': few_shot_metrics,
                'avg_zero_shot': avg_zero_shot, 'avg_few_shot': avg_few_shot}



# 金融问题识别
class FinQCEvaluator:
    dataset = 'finqc'

    zero_shot_prompts = [
        '请根据上下文，分析句子是否为金融问题，选项为是、否，请在这两个选项中选出唯一正确的选项。请仅输出是或否类别，多余文字不要输出。\n\n上下文：{context}\n选项：是、否\n答案：',
        '下面是一段文本，你可以告诉我该文本是金融问题吗？是还是否？你只需要回答是或者否，不需要给出其他内容。\n\n上下文：{context}\n答案：',
        '上下文：{context}\n请根据上下文，判断此文本是否为金融问题，选项为是、否。只需要回答“是”或“否”，不需要给出其他内容。\n答案：'
    ]

    few_shot_prompts = [
        '请根据上下文，分析句子是否为金融问题，选项为是、否，请在这两个选项中选出唯一正确的选项。请遵循以下示例，仅输出是或者否，多余文字不要输出。下面给出了一些样例，按照样例输出答案。\n{context}',
        '下面是一段文本，你可以告诉我该文本是金融问题吗？是还是否？请参考下面的例子进行回答。\n{context}',
        '请根据上下文，判断此文本是否为金融问题，选项为是、否。只需要回答“是”或“否”，不需要给出其他内容。请参考下面的例子进行回答。\n{context}'
    ]

    def __init__(self, cot=False):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    def build_few_shot_prompt(self, prompt, context: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{context}\n选项：是、否\n答案：{answer}'.format(
                context=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{context}\n选项：是、否\n答案：'.format(context=context)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            # 如果需要处理文本，可以调用 clean 函数，例如 if gold == clean(pred):
            if gold == _remove_punctuation(pred):
                s += 1
        return round(s / len(golds) * 100, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], context=example['input'], k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                inputs.append(input_text)
                preds.append(str(llm.generate(input_text)))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, context=example['input'], k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero, model_response_few, {'zero_shot_metrics_finqc': zero_shot_metrics, 'few_shot_metrics_finqc': few_shot_metrics,
                'avg_zero_shot': avg_zero_shot, 'avg_few_shot': avg_few_shot}
# 金融术语
class FinTermEvaluator:
    dataset = 'finterm'

    zero_shot_prompts = [
        '{context}是什么？',
        "可以解释一下{context}这个概念吗？",
        "什么是{context}"
    ]

    def __init__(self, cot=False):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        sentences_1 = golds
        sentences_2 = preds
        model = FlagModel('/root/MyData/data/myModel/bge-large-zh',
                          query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                          use_fp16=True)
        embeddings_1 = model.encode(sentences_1)
        embeddings_2 = model.encode(sentences_2)
        similarity = embeddings_1 @ embeddings_2.T
        sum_ = sum(similarity)
        print(sum_)

        return round(numpy.sum(sum_) / len(similarity), 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))

    def run_evaluation(self, llm):
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        model_response_few = ['no few-shot for finterm']
        return model_response_zero, model_response_few, {'zero_shot_metrics_finterm': zero_shot_metrics, 'avg_zero_shot_finterm': avg_zero_shot}
# 论坛情感分析
class FinFEForumEvaluator:
    dataset = 'finfeforum'

    zero_shot_prompts = [
        '请根据上下文，从股民论坛中提取股民评论所表达的情绪，选项为积极、消极、中性，请在这三个选项中选出唯一正确的选项。请仅输出情绪类别，多余文字不要输出。\n\n上下文：{context}\n选项：积极、消极、中性\n答案：',
        '下面是一段股民论坛中股民的评论，你可以告诉我该评论的情绪倾向是什么吗？积极、消极还是中性？你只需要回答情绪“积极”、“消极”或“中性”，不需要给出其他内容。\n\n上下文：{context}\n答案：',
        '上下文：{context}\n请根据上下文，选出此文本所表现出的情绪，选项为积极、消极、中性。只需要回答情绪“积极”、“消极”或“中性”，不需要给出其他内容。\n答案：',
        #'### 任务描述：<br> 你将阅读一段金融文本。请判断该文本的情绪是：积极情绪、消极情绪、中立情绪。<br> ### 回答规范 ：<br>你的输出内容必须需要从“积极情绪”“消极情绪”“中立情绪”中选择一项尽可能最贴切的。不允许有其他任何输出。注意你只能选择1中情绪判断作为输出。注意不需要输出除了上述选项外的其他任何信息，包括关于做出判断的原因、解释等<br>  ### 金融文本<br>{context}\n答案：'
    ]

    few_shot_prompts = [
        '请根据上下文，从股民论坛中提取股民评论所表达的情绪，选项为积极、消极、中性，请在这三个选项中选出唯一正确的选项。请遵循以下示例，仅输出情绪类别，多余文字不要输出。下面给出了一些样例，按照样例输出答案。\n{context}',
        '下面是一段股民论坛中股民的评论，你可以告诉我该评论的情绪倾向是什么吗？积极、消极还是中性？请参考下面的例子进行回答。\n{context}',
        '请根据上下文，选出此文本所表现出的情绪，选项为积极、消极、中性。只需要回答情绪“积极”、“消极”或“中性”，不需要给出其他内容。请参考下面的例子进行回答。\n{context}'
    ]

    def __init__(self, cot=False):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    def build_few_shot_prompt(self, prompt, context: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{context}\n选项：积极、消极、中性\n答案：{answer}'.format(
                context=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{context}\n选项：积极、消极、中性\n答案：'.format(context=context)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            # 如果需要处理文本，可以调用 clean 函数，例如 if gold == clean(pred):
            if gold == _remove_punctuation(pred):
                s += 1
        return round(s / len(golds) * 100, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], context=example['input'], k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        model_response_zero = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            inputs, golds, preds = [], [], []
            for example in tqdm(self.data):
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, context=example['input'], k=few_shot_k)
                inputs.append(input_text)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        model_response_few = [{'input':i, 'output':o, 'goodanswer':g} for i,o,g in zip(inputs, preds, golds)]
        return model_response_zero, model_response_few, {'zero_shot_metrics_finfeforum': zero_shot_metrics, 'few_shot_metrics_finfeforum': few_shot_metrics,
                'avg_zero_shot': avg_zero_shot, 'avg_few_shot': avg_few_shot}



if __name__ == '__main__':
    pass
