欢迎来到 FinEval 
==========================================

本文介绍了FinEval，由上海财经大学统计与管理学院张立文副教授团队开发，是一个专门为中国金融领域设计的评估基准。FinEval为一系列高质量的选择题，涵盖金融、经济、会计和专业证书四个主题。它包括4661个问题，涵盖34个不同的学术科目。为了确保对模型性能进行全面评估，FinEval采用了各种方法，包括zero-shot，few-shot，Answer Only(AO,仅回答答案)和Chain-of-thought(CoT,思维链提示)。在FinEval上评估最先进的中文和英文大语言模型，结果表明，只有GPT-4在不同的提示设置中达到了70%的准确率，这表明大语言模型在金融领域的增长潜力很大。总体而言，这项研究为未来的大语言模型提供了强大的评估基准，并对其发展局限性提供了宝贵的见解。

您可以在 进阶教程_ 中查看我们的数据集示例，或查看我们的**论文（放链接）**了解更多细节。

.. _开始你的第一步:
.. toctree::
   :maxdepth: 1
   :caption: 开始你的第一步

   get_started/install.md
   get_started/dataset_pre.md
   get_started/quick_start.md

.. _使用说明:
.. toctree::
   :maxdepth: 1
   :caption: 使用说明

   user_guide/how_to_run.md
   user_guide/config.md
   user_guide/api_model.md
   user_guide/custom_model.md
   user_guide/prompt_viewer.md

.. _提示词:
.. toctree::
   :maxdepth: 1
   :caption: 提示词

   prompt/overview.md
   prompt/zero_shot.md
   prompt/few_shot.md
   prompt/cot.md

.. _进阶教程:
.. toctree::
   :maxdepth: 1
   :caption: 进阶教程

   advanced_guide/new_dataset.md
   advanced_guide/new_model.md

.. _其他说明:
.. toctree::
   :maxdepth: 1
   :caption: 其他说明

   other/how_to_submit.md
   other/Contact_Us.md

索引与表格
==================

* :ref:`genindex`
* :ref:`search`
