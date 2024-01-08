# sedimentology_paper_analysis
sedimentology paper classification based on BERT

本项目中的模型基于BERT模型，预训练模型使用pytorch中的[bert-base-uncased](https://huggingface.co/bert-base-uncased)模型，使用地学专家标注文献数据训练，可以识别地学文献中的沉积学学科文献，准确率为XX。

数据来源[Deep-Time Digital Earth](https://ddescholar.acemap.info/)

训练数据格式示例可参考train_data.pkl文件

测试数据范例：test_data.pkl

Requirements:
- python 3.8+
- pytorch 1.12.0
- transformers 4.20.0

训练模型
```python
python BERT_train.py
```
测试模型
```python
python BERT_eval.py
```

模型参数文件：链接：<https://pan.baidu.com/s/18JhOZmOPpwcqLy2yhWDbyA> 提取码：ow4g

测试输出为模型判定的沉积学文献对应的paper_id，根据paper_id可在数据库中调取对应的文献数据。

