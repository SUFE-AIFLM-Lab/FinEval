# Few-shot

## 1. AO（Answer-Only）
AO&few-shot

仅预测答案的few-shot的prompt：它允许模型在有限数量的示例下学习新的类别。相比于Zero-Shot，Few-Shot提供了更多的训练数据，但仍然相对较少。这使得模型能够从少量示例中学习新的类别，并在面对新的输入时进行回答。

```python
以下是中国关于banking_practitioner_qualification_certificate考试的单项选择题，请选出其中的正确答案。

下列关于理财业务的理解，说法有误的是____。
A. 综合理财服务中，银行可以让客户承担一部分风险
B. 与理财顾问服务相比，综合理财服务更强调个性化
C. 私人银行业务除了提供金融产品外，更重要的是提供全面的服务
D. 私人银行业务不是个人理财业务
答案：D

申请个人汽车贷款，借款人应提供一定的担保措施，不包括____。
A. 以贷款所购车辆作抵押
B. 房地产抵押
C. 第三方保证
D. 信用担保
答案：D

下列关于申请个人商用房贷款时借款人须具备的条件表述中，错误的是____。
A. 具有良好的信用记录和还款意愿
B. 具有稳定的收入来源和按时足额偿还贷款本息的能力
C. 已支付所购商用房市场价值30$\%$以上的首付款
D. 具有完全民事行为能力的自然人
答案：C

对于季节性融资，如果某公司在银行有多笔贷款，且贷款可展期，银行一定要确保其不被用于____。
A. 长期投资
B. 股票投资
C. 投机投资
D. 其他投资
答案：A

____，公司信贷可分为固定资产贷款、并购贷款、流动资金贷款。
A. 按贷款经营模式划分
B. 按贷款偿还方式划分
C. 按授信品种划分
D. 按贷款担保方式划分
答案：C

以下四种关于风险概念的理解表述中，错误的是____。
A. 风险是未来结果的不确定性
B. 风险是未来结果(如投资的收益率)对期望的偏离，即波动性
C. 风险是损失的可能性
D. 风险代表了未来损失的大小
答案：
```
