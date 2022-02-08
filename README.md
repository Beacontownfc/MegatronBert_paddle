## 1 简介 

**本项目基于PaddlePaddle复现的MegatronBert，完成情况如下:**

- 在mnli、SQuAD1.1和SQuAD2.0数据集上均达到论文精度
- 我们复现的MegatronBert是基于paddlenlp
- 我们提供aistudio notebook, 帮助您快速验证模型
- 我们提供脚本任务，可以快速使用多卡训练模型

**项目参考：**
- [https://github.com/huggingface/transformers/tree/master/src/transformers/models/megatron_bert](https://github.com/huggingface/transformers/tree/master/src/transformers/models/megatron_bert)


## 2 复现精度
>#### 在SQuAD1.1数据集的测试效果如下表。

|网络 |opt|数据集|EM|EM(原论文)|F1|F1(原论文)
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|MegatronBert|AdamW|SQuAD1.1|88.78|88.0|94.40|94.2|

>复现代码训练日志：
[复现代码训练日志](squad1.1.log)

>
>#### 在SQuAD2.0数据集的测试效果如下表。

|网络 |opt|数据集|EM|EM(原论文)|F1|F1(原论文)
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|MegatronBert|AdamW|SQuAD2.0|85.85|84.8|88.70|88.1|

>复现代码训练日志：
[复现代码训练日志](squad2.0.log)

>#### 在MNLI数据集的测试效果如下表。

|网络 |opt|数据集|acc(matched)|acc(matched)(原论文)|acc(mismatched)|acc(mismatched)(原论文)
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|MegatronBert|AdamW|mnli|89.68|89.7|89.92|90.0|

>复现代码训练日志：
[复现代码训练日志](mnli.log)
>

## 3 数据集
我们主要复现MNLI、SQuAD1.1和SQuAD2.0数据集的精度, 数据集，

SQuAD1.1和SQuAD2.0数据集可以前往此处下载:
[地址](https://aistudio.baidu.com/aistudio/datasetdetail/127315)

MNLI数据集可在此处下载: 
[地址](https://dl.fbaipublicfiles.com/glue/data/MNLI.zip)

或者您也可以不下载数据集，我们已经把数据集处理程序封装好，运行训练脚本会自动下载数据集


## 4环境依赖
运行以下命令即可配置环境
```bash
pip install paddlenlp==2.2.4
```

## 5 快速开始
如果你觉得以下步骤过于繁琐，您可以直接到此处
[链接](https://aistudio.baidu.com/aistudio/projectdetail/3464462)
利用我们提供的AISTUDIO NOTEBOOK快速验证MNLI数据集的结果。

若想快速验证SQuAD1.1和SQuAD2.0数据集的结果,请前往此处:
[链接](https://aistudio.baidu.com/aistudio/projectdetail/3459226)

如果您希望快速训练模型，请前往此处:
[链接](https://aistudio.baidu.com/aistudio/projectdetail/3460019)

`脚本任务说明：`(1)若希望在SQuAD1.1数据集上训练，请使用启动命令`bash run_squad_1.1.sh`

(2)若希望在SQuAD2.0数据集上训练，请使用启动命令`bash run_squad_2.0.sh`

(3)若希望在mnli数据集上训练，请使用启动命令`bash run_mnli.sh`

`请注意，运行bash run_mnli.sh脚本有小概率出现 <文件不存在> 的错误，出现此错误请重启脚本任务`

首先，您需要下载预训练权重:
[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/127287)

###### 训练并测试在mnli数据集上的ACC：


```bash
python -m paddle.distributed.launch run_glue.py --task_name=mnli --output_dir=<OUTPUT_DIR> --model_dir=<MODEL>
```

说明：

- `<OUTPUT_DIR>`和`<MODEL>`分别为输出文件夹路径和预训练权重文件夹路径

运行结束后你将看到如下结果:
```bash
eval loss: 0.186327, acc: 0.8992358634742741, eval loss: 0.332409, acc: 0.8968673718470301, eval done total : 118.65499472618103 s
```

###### 训练并测试在SQuAD1.1数据集上的精度：

```bash
python -m paddle.distributed.launch run_squad.py --do_train --do_predict --model_dir=<MODEL>
```

运行结束后你将看到如下结果:
```bash
{
  "exact": 88.78902554399244,
  "f1": 94.4082803514958,
  "total": 10570,
  "HasAns_exact": 88.78902554399244,
  "HasAns_f1": 94.4082803514958,
  "HasAns_total": 10570
}
```

###### 训练并测试在SQuAD2.0数据集上的精度：

```bash
python -m paddle.distributed.launch run_squad.py --do_train --do_predict --model_dir=<MODEL> --version_2_with_negative
```

运行结束后你将看到如下结果:
```bash
{
  "exact": 85.85867093405206,
  "f1": 88.70579950475263,
  "total": 11873,
  "HasAns_exact": 82.47300944669365,
  "HasAns_f1": 88.17543143048748,
  "HasAns_total": 5928,
  "NoAns_exact": 89.23465096719933,
  "NoAns_f1": 89.23465096719933,
  "NoAns_total": 5945,
  "best_exact": 85.99343047250063,
  "best_exact_thresh": -1.6154582500457764,
  "best_f1": 88.75296534320918,
  "best_f1_thresh": -0.20494508743286133
}
```

## 6 代码结构与详细说明

```
├─args.py                  # SQuAD数据集配置文件
├─modeling_MegatronBERT.py # MegatronBert模型文件
├─run_glue.py              # glue任务训练脚本
├─run_squad.py             # SQuAD任务训练脚本                                 
```