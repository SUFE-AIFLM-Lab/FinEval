# 支持新数据集

## 1. 修改subject_mapping.json

​		以下为`subject_mapping.json`中格式：

	{
		"filename":[
		"英文名称",
		"中文名称"
		"类别"
		]
	}
​	  以下为`subject_mapping.json`中税法例子:

```
{
	"tax_law":[
	"Tax_law",
	"\u7a0e\u6cd5"
	"Accounting"
	]
}
```

如需添加新的数据集及所属类别，按上述格式进行修改`subject_mapping.json`文件

## 2. 确定data数据集格式

新加入数据集由三个部分组成：dev、val 和 test。每个科目的 dev 集包含五个示范实例以及为 few-shot 评估提供的解释。val 集旨在用于超参数调整。而 test 集则用于模型评估。

  以下为会计科目的dev文件(accounting_dev.csv)实例:

  ```
  id,question,A,B,C,D,answer,explanation
  0,甲公司 2×14 年 12 月 20 日与乙公司签订商品销售合同。合同约定：甲公司应于 2×15 年 5 月 20 日前将合同标的商品运抵乙公司并经验收，在商品运抵乙公司前灭失、毁损、价值变动等风险由甲公司承担。甲公司该项合同中所售商品为库存W 商品，2×14 年 12 月 30 日，甲公司根据合同向乙公司开具了增值税专用发票并于当日确认了商品销售收入。W 商品于 2×15 年 5 月 10 日发出并于 5 月 15 日运抵乙公司验收合格。对于甲公司 2×14 年 W 商品销售收入确认的恰当性判断，除考虑与会计准则规定的收入确认条件的符合性以外，还应考虑可能违背的会计基本假设是____。,会计主体,会计分期,持续经营,货币计量,B,题目中明确提到在商品运抵乙公司前灭失、毁损、价值变动等风险由甲公司承担，即相关商品的控制权并未转移，应在2×15年确认收入，甲公司在2×14年确认收入，违背了会计分期假设。
  ```

  以下为会计科目的test文件(accounting_test.csv)实例:

  ```
  id,question,A,B,C,D
  0,下列做法中，不违背会计信息质量可比性要求的有____。,因客户的财务状况好转，将坏账准备的计提比例由应收账款余额的30%降为15%,为了扭转亏损，将本应费用化的借款费用进行资本化,被投资企业本年严重亏损，投资企业将长期股权投资从权益法转为成本法,鉴于本期利润完成不理想，将应费用化的研发支出改成资本化处理
  ```

  以下为会计科目的val文件(accounting_val.csv)实例:

  ```
  id,question,A,B,C,D,answer
  0,甲公司在非同一控制下企业合并中取得 10 台生产设备，合并日以公允价值计量这些生产设备。甲公司可以进入 X 市场或 Y 市场出售这些生产设备，合并日相同生产设备每台交易价格分别为 180 万元和 175 万元。如果甲公司在 X 市场出售这些合并中取得的生产设备，需要支付相关交易费用 100 万元，将这些生产设备运到 X 市场需要支付运费 60 万元。如果甲公司在 Y 市场出售这些合并中取得的生产设备，需要支付相关交易费用 80 万元，将这些生产设备运到 Y 市场需要支付运费 20 万元。假定上述生产设备不存在主要市场，不考虑增值税及其他因素，甲公司上述生产设备的公允价值总额是____。,1640万元,1650万元,1730万元,1740万元,C
  ```