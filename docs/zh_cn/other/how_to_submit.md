# 如何提交您的测评

测评结果保存位置为：`output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date` ，该文件夹下自动生成`submission.json`，用户只需提交该文件。

在[如何运行](https://fineval.readthedocs.io/zh_CN/latest/user_guide/how_to_run.html)中有保存位置的说明。

`submission.json`文件格式如下：


```text
## 每个学科内部的键名是数据集中的"id"字段
{
    "banking_practitioner_qualification_certificate": {
        "0": "A",
        "1": "B",
        "2": "B",
        ...
    },
    
    "学科名称":{
    "0":"答案1",
    "1":"答案2",
    ...
    }
    ....
}
```


您可以将生成的`submission.json`文件以邮件形式提交到zhang.liwen@shufe.edu.cn。
