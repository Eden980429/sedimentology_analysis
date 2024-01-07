import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook
from transformers import AdamW
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertModel, BertConfig
import elasticsearch
import random

"""
data preparation
"""

index = "dde_deepdive"
es = elasticsearch.Elasticsearch("10.10.10.10:9200", timeout=1000)

def search_es_count(es,index,pids):   
    query_json = {
        "size" : 4000000,
        "_source":{
            "includes": [
            ]
        },
        
        'query': {
                'bool': {                   
                    "must": [{
                                 "terms":{"am_id": pids}
                             }
                    ],
                }
            }

    }
    query = es.search(index=index, body=query_json)
    
    doc = []
    ids = []
    for item in query['hits']['hits']:
        text = item['_source']['title'] + item['_source']['abstract']
        doc.append(text)
        ids.append(item['_source']['am_id'])
        if len(text) < 50:
            print(item['_source']['am_id'], '\n', text)
    
    return {'doc':doc,'pids':ids}

def get_saved_id(filename):
    ids = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            ids.append(line)
            
    return ids

def get_saved_id_pkl(filename):
    ids = []
    with open(filename, 'rb') as f:
        ids = pickle.load(f)
    return ids

def load_variable(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

test=load_variable('dataset0817/test0817_3journal.pkl')
test_text=[]
test_label=[]
for i in test:
    test_label.append(i[0])
    test_text.append(i[1])

"""
model test
"""
class BertForAffiliationNameNormalization(torch.nn.Module):
    
    def __init__(self, num_of_classes):
        super(BertForAffiliationNameNormalization, self).__init__()
        self.num_of_classes = num_of_classes
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.dropout = nn.Dropout(p=0.1, inplace=False).to(device)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_of_classes, bias=True).to(device)
        
        
    def forward(self, input_ids, attention_mask):
        pooled_out = self.bert(input_ids, attention_mask=attention_mask)
        pooled_out = self.dropout(pooled_out[1])
        logits = self.classifier(pooled_out)
        
        return logits

class SediDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = BertForAffiliationNameNormalization(2)
model = nn.DataParallel(model, device_ids=[0,2,3])

# 这里用你需要测试的checkpoint，一般为最后一个即可
model=torch.load('checkpoint5/After_epoch_9_bert.pkl')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

test_encodings = tokenizer(test_text, truncation=True, padding=True)
test_dataset = SediDataset(test_encodings, test_label)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

len(test_loader)

def sigmoid(x): 
    return 1/(1 + np.exp(-x))

def predict(logits,threshold):
    result=[]
    pre=sigmoid(logits.detach().cpu().numpy())
    
    for i in pre:
        if i[1]>=threshold:
            result.append(1)
        else: result.append(0)
    
    return result

def P_R(threshold, mark_label, mark_predi):
    TP=0 #label=1, pre=1
    FP=0 #label=0, pre=1
    FN=0 #label=1, pre=0
    TN=0 #label=0, pre=0
    
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)   
        logits = model(input_ids, attention_mask=attention_mask)
        label = labels.to('cpu').numpy()
#         print('label', label)
        #print(type(label))
        mark_label.append(label)
        
        predi = predict(logits,threshold)
        mark_predi.append(predi)
        for i in range(len(label)):
            if label[i] == 1:
                if predi[i] == 1:
                    TP += 1
                else: FN += 1
            elif predi[i] == 1:
                FP += 1
            else: TN += 1
                

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*(precision*recall)/(precision+recall)
#     precision = 1
#     recall = 1
#     F1 = 1
    
    #return precision, recall
    return TP, FP, FN, TN, precision, recall, F1, mark_label, mark_predi
    
#不同阈值下，准确率/召回率
# F1 = 2*(precision*recall)/(precision+recall)
mark_label = []
mark_predi = []
for threshold in np.arange(0.1,0.8,0.05):
    print('threshold', threshold)
    tmp_label = []
    tmp_predi = []
    TP, FP, FN, TN, precision, recall, F1, tmp_label, tmp_predi = P_R(threshold, tmp_label, tmp_predi)
    if threshold == 0.4:
        mark_label = tmp_label
        maek_predi = tmp_predi
    print(TP, FP, FN, TN, precision, recall, F1)