import os
from tqdm import tqdm
import jieba
# from FlagEmbedding import FlagModel
from utils import _remove_punctuation, _mixed_segmentation, _find_lcs, write_json, load_json, \
    extract_questions_and_text, _compute_f1_score, compute
import numpy
import json

DATA_PATH = 'data'

from openai import OpenAI
import os

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")

    api_key="sk-3wWb7Jy5NVMiVrLp22919e4e51A64fF29e37030373392bA6",
    base_url="https://api.132999.xyz/v1"
)


# api_key="sk-cSfer8keII0iCp8VpjBio1PC1AK4hcVv4NMrpYE03wpk0gEE",
# base_url="https://api.chatanywhere.tech"

# api_key="sk-3wWb7Jy5NVMiVrLp22919e4e51A64fF29e37030373392bA6",
#   base_url="https://api.132999.xyz/v1"

def get_completion(prompt, model="gpt-4"):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        temperature=0,  # this is the degree of randomness of
    )
    return response.choices[0].message.content


def recognize_and_convert(input_string):
    try:
        number = int(input_string)
        if 0 <= number <= 10:
            return number
        else:
            return input_string
    except ValueError:
        return input_string


# 任务分解
class FinTASK:
    dataset = 'fintask'

    zero_shot_prompts = [
        '你是一个金融领域的专家，请对我给你的金融任务进行分解与规划。\n金融任务：' + '{instruction}',
    ]

    few_shot_prompts = [
        '你是一个金融领域的专家，请对我给你的金融任务进行分解与规划。\n下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                instruction = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                print("模型回答：\n", tmp)
                print(example['goodanswer'])
                output.append({'input': example['input'], 'output:': tmp, 'goodanswer': example['goodanswer']})
        return output

    def evaluate_gpt4(self, output_llm_dir):
        # 用gpt4来对存储好的模型回答评分
        llm_output = load_json(os.path.join(output_llm_dir, self.dataset + '-output.json'))
        scores = []
        scores_all = []
        for example in tqdm(llm_output):
            inputt = example['input']
            output = example['output:']
            goodanswer = example['goodanswer']

            instruction = "我会给你一个'问题'，与一个'待打分回答'，请根据评分标准对'待打分回答'进行打分。\n# 回复要求：你只需要回复一个数字表示总分，如'2',不需要具体的评分过程。请不要轻易给5分。\n## 评分标准：1. 完整性：任务规划完全覆盖了所有关键点和必要的细节。加1分。\n2. 准确性：对于任务规划要求的理解完全准确，无错误信息。加1分。\n3. 逻辑性和条理性：结构清晰，逻辑连贯，易于理解。加1分。\n4. 实用性和可行性：提出的方案或信息非常实用，具有高度的可行性。加1分。\n5. 创新性：提供了创新性的见解或独特的解决方案。加1分。\n6. 满分5分，你只有0，1，2，3，4，5六个选项。\n## 问题：\n{}\n## 待打分回答：\n{}\n#回复要求：你只需要回复一个数字表示总分，如'2'，不需要具体的评分过程。严格按照评分标准打分。请不要轻易给5分\n"
            instruction = instruction.format(inputt, output)
            sc = get_completion(instruction)
            print(sc)
            sc = recognize_and_convert(sc)  # 转成int
            scores_all.append(sc)
            if isinstance(sc, str):
                continue
            scores.append(sc)
        ave_score = sum(scores) / len(scores)
        return scores_all, ave_score


# 思维链
class FinCoT:
    dataset = 'fincot'

    zero_shot_prompts = [
        '你是一个金融领域的专家，请按照如下格式分步骤解决一个金融领域的题目' + '解题步骤\n1)...\n2)...\n3)...\n所以，这里是这道题目的答案,请输出题目最终的数字答案。' + '{instruction}',
    ]

    few_shot_prompts = [
        '你是一个金融领域的专家，请按照如下格式分步骤解决一个金融领域的题目。' + '解题步骤\n1)...\n2)...\n3)...\n所以，这里是这道题目的答案\n' + '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                instruction = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                print("模型回答：\n", tmp)
                print(example['goodanswer'])
                output.append({'input': example['input'], 'output:': tmp, 'goodanswer': example['goodanswer']})
        return output

    def evaluate_gpt4(self, output_llm_dir):
        # 用gpt4来对存储好的模型回答评分
        llm_output = load_json(os.path.join(output_llm_dir, self.dataset + '-output.json'))
        scores = []
        scores_all = []
        for example in tqdm(llm_output):
            inputt = example['input']
            output = example['output:']
            goodanswer = example['goodanswer']

            instruction = "我会给你一个'题目'，一个'正确回答'与一个'待打分回答'，请根据评分标准对'待打分回答'进行打分。\n# 回复要求：你只需要回复一个数字表示总分，如'2'或'1' , 不需要具体的评分过程。严格按照评分标准打分。\n## 评分标准：\n1. 如果语言完整通顺且与问题相关，加1分\n2. 如果解题过程中含有数学推理并且最后最后一行给出了最终的答案（必须是数字答案），加1分\n3. 正确答案在'正确回答'的最后一行，如果回答最后得出了正确答案，加3分\n4. 满分共5分，你只有0，1，2，5四个选项\n5.只有最终答案与正确答案一样才能给5分\n## 题目：\n{}\n## 正确回答：\n{}\n## 待打分回答：\n{}\n# 回复要求：你只需要回复一个数字表示总分，如'2'或'1'，不需要具体的评分过程。严格按照评分标准打分。\n"
            instruction = instruction.format(inputt, goodanswer, output)
            sc = get_completion(instruction)
            sc = recognize_and_convert(sc)  # 转成int
            print(sc)
            scores_all.append(sc)
            if isinstance(sc, str):
                continue
            scores.append(sc)
        ave_score = sum(scores) / len(scores)
        return scores_all, ave_score


# 检索增强
class FinRAG:
    dataset = 'finrag'

    zero_shot_prompts = [
        '{instruction}',
    ]

    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                instruction = example['instruction']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                print("模型回答：\n", tmp)
                print(example['output'])
                output.append({'input': example['instruction'], 'output:': tmp, 'goodanswer': example['output']})
        return output

    def evaluate_gpt4(self, output_llm_dir):
        # 用gpt4来对存储好的模型回答评分
        llm_output = load_json(os.path.join(output_llm_dir, self.dataset + '-output.json'))
        scores = []
        scores_all = []
        for example in tqdm(llm_output):
            inputt = example['input']
            output = example['output:']
            goodanswer = example['goodanswer']

            instruction = "我会给你一个'题目'，一个'正确回答'与一个'待打分回答'，请根据评分标准对'待打分回答'进行打分。\n# 回复要求：你只需要回复一个数字表示总分，如'2'或'3' , 不需要具体的评分>过程。严格按照评分标准打分。\n## 评分标准：\n1. 如果回答准确，与正确回答含义一致，加2分\n2. 如果回答完整，覆盖了正确回答的所有关键点，加1分\n3. 如果回答高效，无冗余信息，并且在规定字数内，加1分\n4. 如果回答具有正确回答没有但可以从材料中推理出的合理的信息，加1分\n5. 满分共5分，你只有0，1，2，3，4, 5六个选项\n## 题目：\n{}\n## 正确回答：\n{}\n## 待打分回答：\n{}\n# 回复要>求：你只需要回复一个数字表示总分，如'2'或'3'，不需要具体的评分过程。严格按照评分标准打分。\n"
            instruction = instruction.format(inputt, goodanswer, output)
            sc = get_completion(instruction)
            sc = recognize_and_convert(sc)  # 转成int
            print(sc)
            scores_all.append(sc)
            if isinstance(sc, str):
                continue
            scores.append(sc)
        ave_score = sum(scores) / len(scores)
        return scores_all, ave_score


##########################################################################################################################################

class FinDiag:
    dataset = 'findiag'

    zero_shot_prompts = [
        '{instruction}',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                instruction = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                print("模型回答：\n", tmp)
                print(example['goodanswer'])
                output.append({'input': example['input'], 'output:': tmp, 'goodanswer': example['goodanswer']})
        return output

    def evaluate_gpt4(self, output_llm_dir):
        # 用gpt4来对存储好的模型回答评分
        llm_output = load_json(os.path.join(output_llm_dir, self.dataset + '-output.json'))
        scores = []
        scores_all = []
        for example in tqdm(llm_output):
            inputt = example['input']
            output = example['goodanswer']
            goodanswer = example['goodanswer']

            instruction = "1. 背景介绍:我们希望测评agent长程对话能力：在现实世界中的长对话中，用户通常会使用大模型谈论几个话题并在其中切换。比如主题检索任务，是通过要求大模型检索由多个主题组成的长对话中的开头和中间过程的主题来测试这种场景。2. 任务介绍: 我们已经针对该领域向各个agent提出了一系列问题，并收集到了相应答案。具体见“4.提问与回答”小节。其中，提问为我们给出的问题，回答为被测评agent给出的答案。 请严格依据“3.评分标准”，分析问题及参考答案，根据评分标准对答案进行评分。请逐项列出每一项的得分，并用简短的一句话陈述评分依据。 请分别给出各项得分后，最终进行加总。3. 评分标准 在针对金融领域API调用能力评分时，我们希望测评agent长程对话能力：用户通常会使用大模型谈论几个话题并在其中切换。比如主题检索任务，是通过要求大模型检索由多个主题组成的长对话中的开头和中间过程的主题来测试这种场景。0分：如果回答完全无关或误导用户，没有提供任何对用户询问有用的信息。这表明智能体没有理解用户的问题，或者给出了一个完全脱离主题的答案。1分：如果回答虽然与用户的查询相关，但信息非常有限、片面或包含大量不相关的内容。这表示智能体对问题有基本的理解，但提供的信息对用户几乎没有帮助。2分：如果回答解决了用户问题的一部分，提供了一些有价值的信息，但并未全面覆盖或直接回应用户的所有细节和要求。这意味着智能体抓住了问题的一部分要点，但回答缺乏深度或广度。3分：如果回答明确回应了用户问题的基本要素，并提供了实用的信息，无论其表达方式是更倾向于AI辅助形式还是似乎借鉴了博客或搜索结果的元素。这表示智能体能够理解并针对用户的问题提供一个具体且有用的答案。4分：如果回答直接、全面地从AI辅助的角度出发，明确回应了用户的所有问题，且内容组织良好、有帮助。即使在清晰度、简洁性或焦点上略有提升空间，也显示出智能体的高度理解和适当的回答结构。5分：对于完美对用户的问题进行了量身定制的回答，不含任何无关信息，反映出专家级知识，展示了高质量、引人入胜且富有洞察力的答案。这代表智能体不仅深入理解用户的需求，还能够以精确、专业的方式提供答案，充分展示了其在金融领域的专业能力和深度分析技巧。4. 提问与回答 提问{}. 回答{}5. 注意事项：1.请忽略“追问”部分的内容，仅对第一个问题的“回答”进行评分。2.请仅输出最终的打分结果，忽略掉中间的解释和分析过程。我只需要一个int或float的数字，如2"

            instruction = instruction.format(inputt, output)

            sc = get_completion(instruction)
            print(sc)
            sc = recognize_and_convert(sc)  # 转成int
            scores_all.append(sc)
            if isinstance(sc, str):
                continue
            scores.append(sc)
        ave_score = sum(scores) / len(scores)
        return scores_all, ave_score


class FinDoc:
    dataset = 'findoc'

    zero_shot_prompts = [
        '{instruction}',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                instruction = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                print("模型回答：\n", tmp)
                print(example['goodanswer'])
                output.append({'input': example['input'], 'output:': tmp, 'goodanswer': example['goodanswer']})
        return output

    def evaluate_gpt4(self, output_llm_dir):
        # 用gpt4来对存储好的模型回答评分
        llm_output = load_json(os.path.join(output_llm_dir, self.dataset + '-output.json'))
        scores = []
        scores_all = []
        for example in tqdm(llm_output):
            inputt = example['input']
            output = example['output:']
            goodanswer = example['goodanswer']

            instruction = "1. 背景介绍:我们希望测评agent在金融领域的多文档问答能力, 请你根据评分标准打分。2. 任务介绍: 在“4.提问与回答”小节中，提问为我们给出的问题，回答为被测评agent给出的答案。 请严格依据“3.评分标准”，分析问题及给出的答案，根据评分标准对答案进行评分。请一步一步思考每一项的得分，在分别给出各项得分后，最终进行准确加总。3. 评分标准 以下是基于十分制，综合考量各方面因素的评分标准：1. 相关性（0-2分）0分：回答与提问主题完全无关。1分：回答提供了一些相关信息，但包含不少不相关的内容。2分：回答紧密相关，完全针对用户的提问，无不相关信息。2. 信息的完整性和准确性（0-3分）0分：回答中的信息严重缺失或错误，未能准确回答用户的问题。1分：回答包含一部分正确信息，但遗漏了一些关键点或包含不准确的信息。2分：回答基本完整和准确，涵盖了大部分用户提问的要点，但可能存在少量遗漏或小的不准确之处。3分：回答信息完整、准确无误，全面覆盖了用户提问的所有方面。3. 组织与表达（0-2分）0分：回答组织混乱，难以理解，有严重的语言错误。1分：回答结构基本合理，但在清晰度或语言流畅性方面有待提高。2分：回答结构清晰，逻辑严谨，表达流畅，容易理解。4. 依据提供和引用准确性及其正确应用（0-3分）0分：没有提供任何文档引用，或者引用不正确，且陈述基于额外信息而不是从提供的文本中学习到的。1分：尽管尝试引用文档，部分引用有误，或对文档的理解和应用部分正确。2分：正确地引用了一部分相应文档，但在全面性或准确性方面仍有改进空间。3分：正确且准确地引用了相应文档，并且仅基于这些文档的内容构建了回答，没有添加任何额外的、未经证实的信息。4. 提问与回答 提问{}. 回答{} 5. 注意事项：1.若出现“追问”部分的内容请忽略，仅对第一个问题的“回答”进行评分,若无追问则直接正常评分即可。2.请仅输出最终的打分结果，忽略掉中间的解释和分析过程。我只需要一个int或float的数字，如2"

            instruction = instruction.format(inputt, output)

            sc = get_completion(instruction)
            print(sc)
            sc = recognize_and_convert(sc)  # 转成int
            scores_all.append(sc)
            if isinstance(sc, str):
                continue
            scores.append(sc)
        ave_score = sum(scores) / len(scores)
        return scores_all, ave_score


class APIUtil:
    dataset = 'apiutil'

    zero_shot_prompts = [
        '{instruction}',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                instruction = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                print("模型回答：\n", tmp)
                print(example['goodanswer'])
                output.append({'input': example['input'], 'output:': tmp, 'goodanswer': example['goodanswer']})
        return output

    def evaluate_gpt4(self, output_llm_dir):
        # 用gpt4来对存储好的模型回答评分
        llm_output = load_json(os.path.join(output_llm_dir, self.dataset + '-output.json'))
        scores = []
        scores_all = []
        for example in tqdm(llm_output):
            inputt = example['input']
            output = example['output:']
            goodanswer = example['goodanswer']

            instruction = "在“4.提问与回答”中，提问为给出的问题，回答为被测评agent的答案。 请严格依据“3.评分标准”分析问题并根据评分标准对回答进行评分。请逐项评价每一项的得分，但不要进行输出！请最终进行加总并给我最后的分数。3. 评分标准 以下是基于十分制的评分标准：理解和规划（2分）:分配0分，如果Agent没有表现出对提问API描述的基本理解；或Agent没有明确回答。分配0.5分，如果Agent表现出了对提问API描述的基本理解，但理解有一定偏差。分配1分，如果Agent不仅基本正确理解API描述，还能明确指出所需的API功能（如数据检索、分析）。分配2分，如果Agent能够精确地规划出如何使用API完成任务，包括理解API的输入、输出和功能限制。代码实现（3分）:分配0分，如果Agent基本没有给出任何有用的代码实现或存在许多错误遗漏。或有大量操作Agent只进行描述未实际给出代码。分配1分，如果Agent提供了部分相关的代码实现但存在一些错误或遗漏；分配2分，如果Agent的调用的api全部正确只偶尔不完整或有小错误。分配3分，如果AI Agent提供了完整的、正确的代码实现，包括准确调用API并合理处理API返回的数据。数据处理与分析（2分）:分配0分，如果Agent基本无法处理和分析API返回的数据或分析完全欠缺深度或广度；或有大量操作只进行描述未实际给出操作方法；分配1分，如果Agent能基本处理和分析API返回的数据但分析深度或广度不足。分配2分，如果Agent展示了高级的数据处理和分析技巧，能从数据中提取出有深度的洞见。结果准确性和完整性（2分）:分配0分，如果Agent返回的结果基本是错误或不可接受的，或没有完整具体完成任务（包括通过分析跳过具体实现等）。分配1分，如果Agent返回的结果基本正确，但缺少关键细节或部分不准确。分配2分，如果Agent的回答完全准确，没有遗漏任何关键信息。清晰度和表达（1分）:分配0分，如果如果Agent的回答基本不具有可读性。分配0.5分，如果Agent的回答可读，但可能存在一些语言上的不清晰或组织上的问题；或回答过于冗长、啰嗦，存在不必要的信息分配1分，如果Agent的回答语言表达清晰，逻辑组织良好，易于理解。4. 提问与回答 提问{}. 回答{} 5. 注意事项：(1)若出现“追问”部分的内容请忽略，仅对第一个问题的“回答”进行评分，不需要有任何说明。(2)在评分时，你需要认为输出答案的使用用户对于该领域或编程方面完全没有任何了解。因此用户将严格按照给定的回答进行操作。故如果给出的指令并不能真正直接运行，该回答应当认为是不可接受的。agent给出的指令应当是清晰、具体、完整、可执行的。(3)请仅输出最终的打分结果，忽略掉中间的解释和分析过程。我只需要一个int或float的数字，如2。在输出前请确保你只返回了数字。"

            instruction = instruction.format(inputt, output)

            sc = get_completion(instruction)
            print(sc)
            sc = recognize_and_convert(sc)  # 转成int
            scores_all.append(sc)
            if isinstance(sc, str):
                continue
            scores.append(sc)
        ave_score = sum(scores) / len(scores)
        return scores_all, ave_score


class APIFind:
    dataset = 'apifind'

    zero_shot_prompts = [
        '{instruction}',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                instruction = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                print("模型回答：\n", tmp)
                print(example['goodanswer'])
                output.append({'input': example['input'], 'output:': tmp, 'goodanswer': example['goodanswer']})
        return output

    def evaluate_gpt4(self, output_llm_dir):
        # 用gpt4来对存储好的模型回答评分
        llm_output = load_json(os.path.join(output_llm_dir, self.dataset + '-output.json'))
        scores = []
        scores_all = []
        for example in tqdm(llm_output):
            inputt = example['input']
            output = example['output:']
            goodanswer = example['goodanswer']

            instruction = "在“4.提问与回答”中，提问为给出的问题，回答为被测评agent的答案。 请严格依据“3.评分标准”分析问题并根据评分标准对回答进行评分。请逐项评价每一项的得分，但不要进行输出！请最终进行加总并给我最后的分数。3. 评分标准 以下是基于十分制的评分标准：1. 任务相关性和信息完整性（最高3分）0分: 回答与提出的问题完全不相关，没有提供任何有用信息。1分: 回答虽然尝试相关，但信息量很少，内容大部分与任务无关。2分: 回答相关并提供了部分所需信息，但没有完全覆盖用户的全部需求。3分: 回答完全相关，全面且准确地提供了用户所需的全部信息。2. 参数选择和调用逻辑（最高3分）0分: 完全没有选择正确的API参数，提供的调用逻辑完全错误或缺失。1分: 选择了一些正确的API参数，但调用逻辑存在较大误差。2分: 绝大多数API参数选择正确，调用逻辑基本合理，但存在小的瑕疵。3分: API参数选择完全正确，调用逻辑清晰且完全合乎需求。3. 语言的通顺性（最高2分）0分: 回答难以理解，语言表达混乱，有大量语病。1分: 回答的语言表达存在小的错误或不够流畅，但不影响理解。2分: 回答的语言表达通顺流畅，没有语病，非常易于理解。4. 综合理解和细节处理能力（最高2分）0分: AI Agent表现出对问题的综合理解非常有限，忽略了关键细节，解决方案缺乏深度和准确性。1分: AI Agent对问题有一定程度的综合理解，能够识别出一些关键细节并进行处理，但仍有遗漏或处理不够精准的地方。2分: AI Agent展现出优秀的综合理解能力，能够准确捕捉并处理问题的所有关键细节，提供的解决方案既全面又具体，展示了高水平的细节关注和处理能力。 4. 提问与回答 提问{}. 回答{} 5. 注意事项：(1)在评分时，你需要认为输出答案的使用用户对于该领域或编程方面完全没有任何了解。因此用户将严格按照给定的回答进行操作。故如果给出的指令并不能真正直接运行，该回答应当认为是不可接受的。agent给出的指令应当是清晰、具体、完整、可执行的。(2)请仅输出最终的打分结果，忽略掉中间的解释和分析过程。我只需要一个int或float的数字，如2。在输出前请确保你只返回了数字。"

            instruction = instruction.format(inputt, output)

            sc = get_completion(instruction)
            print(sc)
            sc = recognize_and_convert(sc)  # 转成int
            scores_all.append(sc)
            if isinstance(sc, str):
                continue
            scores.append(sc)
        ave_score = sum(scores) / len(scores)
        return scores_all, ave_score


# Application Security
class AppSafe:
    dataset = 'appsafe'

    zero_shot_prompts = [
        '下面是关于application security的单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class CrypSafe:
    dataset = 'crypsafe'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在>输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class MalWare:
    dataset = 'malware'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class MemSafe:
    dataset = 'memsafe'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在>输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class NetwrkSafe:
    dataset = 'netwrksafe'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在>输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class PenTest:
    dataset = 'pentest'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在>输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class RevEng:
    dataset = 'reveng'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在>输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class SftwrSafe:
    dataset = 'sftwrsafe'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class SysSafe:
    dataset = 'syssafe'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在>输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class Vulnrb:
    dataset = 'vulnrb'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在>输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output


class WebSafe:
    dataset = 'websafe'

    zero_shot_prompts = [
        '下面是单项选择问题。你将收到一个问题和四个选项：A、B、C、D。请从这四个选项中选择最正确贴切的回答。注意，你只需要回答A/B/C/D中的一个字母即可，不需要回答完整的选项，更不允许输出解释或其他无关输出。题目如下：{instruction}。请在输出前再次检查格式是否正确，你只允许输出ABCD四个字母中的一个。不要输出任何其他内容。请在>输出一个字母后立即停止输出。',
    ]

    # Few-shot not for this task
    few_shot_prompts = [
        '下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
    ]

    def __init__(self):
        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.json'))

    @staticmethod
    def build_zero_shot_prompt(prompt, inst):
        return prompt.format(instruction=inst)

    def run_evaluation(self, llm):
        # 模型对该数据集所有问题输出的回答
        output = []
        score = 0
        count = 0
        for zero_shot_prompt in self.zero_shot_prompts:
            for example in tqdm(self.data):
                question = example['question'] + 'A' + example['A'] + 'B' + example['B'] + 'C' + example['C'] + 'D' + \
                           example['D']
                instruction = question
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
                tmp = llm.generate(input_text)
                # print("模型回答:",tmp)
                # print(f"标准答案:{example['answer']}")
                count += 1
                if tmp == example['answer']:
                    score += 1
                # print(f"score:{score}")
                output.append(
                    {'input': question, 'output': tmp, 'goodanswer': example['answer'], 'accScore': score / count})
        print(f"Total Sc of {self.dataset}: {score / count}")
        return output

    # def run_evaluation(self, llm):
    #     #模型对该数据集所有问题输出的回答
    #     output = []
    #     score = 0
    #     count = 0
    #     for zero_shot_prompt in self.zero_shot_prompts:
    #         for example in tqdm(self.data):
    #             question = example['question']+'A'+example['A']+'B'+example['B']+'C'+example['C']+'D'+example['D']
    #             instruction = question
    #             input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inst=instruction)
    #             tmp = llm.generate(input_text)
    #             print("模型回答:",tmp)
    #             print(f"标准答案:{example['answer']}")
    #             count += 1
    #             if tmp==example['answer']:
    #                 score += 1
    #             print(f"score:{score}")
    #             output.append({'input':question,'output':tmp, 'goodanswer':example['answer'],'score':score})
    #     print(f"Total Sc:{score/count}")
    #     return output
