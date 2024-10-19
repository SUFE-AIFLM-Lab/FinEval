# Add a dataset

## 1. Modify subject_mapping.json

​		The following is the format in `subject_mapping.json`：

	{
		"filename":[
		"English name",
		"Chineses name"
		"categories"
		]
	}
​	  The following is an example of tax law in `subject_mapping.json`:

```text
{
	"tax_law":[
	"Tax_law",
	"\u7a0e\u6cd5"
	"Accounting"
	]
}
```

If you need to add a new dataset and its category, modify the `subject_mapping.json` file according to the above format

## 2. Determine the format of the data dataset

The newly added dataset consists of three parts: dev, val and test. The dev set for each subject contains five demonstration instances with explanations provided for the few-shot evaluation. The val set is intended for hyperparameter tuning. The test set is used for model evaluation.

  The following is an example of the dev file (accounting_dev.csv) of the accounting subject:

  ```
  id,question,A,B,C,D,answer,explanation
  0,甲公司 2×14 年 12 月 20 日与乙公司签订商品销售合同。合同约定：甲公司应于 2×15 年 5 月 20 日前将合同标的商品运抵乙公司并经验收，在商品运抵乙公司前灭失、毁损、价值变动等风险由甲公司承担。甲公司该项合同中所售商品为库存W 商品，2×14 年 12 月 30 日，甲公司根据合同向乙公司开具了增值税专用发票并于当日确认了商品销售收入。W 商品于 2×15 年 5 月 10 日发出并于 5 月 15 日运抵乙公司验收合格。对于甲公司 2×14 年 W 商品销售收入确认的恰当性判断，除考虑与会计准则规定的收入确认条件的符合性以外，还应考虑可能违背的会计基本假设是____。,会计主体,会计分期,持续经营,货币计量,B,题目中明确提到在商品运抵乙公司前灭失、毁损、价值变动等风险由甲公司承担，即相关商品的控制权并未转移，应在2×15年确认收入，甲公司在2×14年确认收入，违背了会计分期假设。
  ```

  The following is an example of the test file (accounting_test.csv) for accounting subjects:
  ```
  id,question,A,B,C,D
  0,下列做法中，不违背会计信息质量可比性要求的有____。,因客户的财务状况好转，将坏账准备的计提比例由应收账款余额的30%降为15%,为了扭转亏损，将本应费用化的借款费用进行资本化,被投资企业本年严重亏损，投资企业将长期股权投资从权益法转为成本法,鉴于本期利润完成不理想，将应费用化的研发支出改成资本化处理
  ```

  The following is an example of the val file (accounting_val.csv) of the accounting subject:

  ```
  id,question,A,B,C,D,answer
  0,甲公司在非同一控制下企业合并中取得 10 台生产设备，合并日以公允价值计量这些生产设备。甲公司可以进入 X 市场或 Y 市场出售这些生产设备，合并日相同生产设备每台交易价格分别为 180 万元和 175 万元。如果甲公司在 X 市场出售这些合并中取得的生产设备，需要支付相关交易费用 100 万元，将这些生产设备运到 X 市场需要支付运费 60 万元。如果甲公司在 Y 市场出售这些合并中取得的生产设备，需要支付相关交易费用 80 万元，将这些生产设备运到 Y 市场需要支付运费 20 万元。假定上述生产设备不存在主要市场，不考虑增值税及其他因素，甲公司上述生产设备的公允价值总额是____。,1640万元,1650万元,1730万元,1740万元,C
  ```
