# sedimentology_analysis
paper classification based on BERT

本项目中的模型使用了BERT模型，使用地学专家标注文献数据训练，可以识别地学文献中的沉积学学科文献，准确率为。

训练

python BERT_train.py

测试

python BERT_eval.py

训练数据集示例可参考文件：

测试输出为文献对应的编号，可根据编号在数据库中调取对应的文献数据。

