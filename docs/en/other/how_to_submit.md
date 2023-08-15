# How to Submit Your Assessment

The location for saving the assessment results is: `output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date`. Within this folder, the `submission.json` file is generated automatically. Users only need to submit this file.

Instructions for the saving location can be found in the [How to run](/docs/en/user_guide/how_to_run.md) section.

```text
## The key inside each subject is the "id" field in the dataset
{
    "banking_practitioner_qualification_certificate": {
        "0": "A",
        "1": "B",
        "2": "B",
        ...
    },
    
    "Subject Name":{
    "0":"Answer1",
    "1":"Answer2",
    ...
    }
    ....
}
```
Then you can submit the prepared JSON file to zhang.liwen@shufe.edu.cn.
