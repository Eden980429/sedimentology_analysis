import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook
from transformers import AdamW
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertModel, BertConfig

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def save_variable(v,filename):
  f=open(filename,'wb')
  pickle.dump(v,f)
  f.close()
  return filename

def load_variable(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

"""
data preparation
"""
# train 训练集；test 测试集；valid 验证集，一般按 8:1:1划分
# train/test/valid_text为输入的文本信息，为title + abatrct的字符串
# train/test/valid_text为标签，是沉积学：1，非沉积学：0
# 按需求改这一部分

train=load_variable('dataset0817/train0817.pkl')
train_texts=[]
train_labels=[]
for i in train:
    train_labels.append(i[0])
    train_texts.append(i[1])
# print(train_labels)
# print(train_texts)

test=load_variable('dataset0817/test0815.pkl')
test_texts=[]
test_labels=[]
for i in test:
    test_labels.append(i[0])
    test_texts.append(i[1])

valid=load_variable('dataset0817/test0815.pkl')
valid_texts=[]
valid_labels=[]
for i in valid:
    valid_labels.append(i[0])
    valid_texts.append(i[1])

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(valid_texts, truncation=True, padding=True)
#test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

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

train_dataset = SediDataset(train_encodings, train_labels)
val_dataset = SediDataset(val_encodings, valid_labels)
#test_dataset = SediDataset(test_encodings, test_labels)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=True)

"""
model training
"""
class BertForAffiliationNameNormalization(torch.nn.Module):
    
    def __init__(self, num_of_classes):
        super(BertForAffiliationNameNormalization, self).__init__()
        self.num_of_classes = num_of_classes
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        print('self.bert ', self.bert.device)
        self.dropout = nn.Dropout(p=0.1, inplace=False).to(device)
#         print('self.dropout ', self.dropout.device)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_of_classes, bias=True).to(device)
#         print('self.classifier ', self.classifier.device)
        
        
    def forward(self, input_ids, attention_mask):
        pooled_out = self.bert(input_ids, attention_mask=attention_mask)
        pooled_out = self.dropout(pooled_out[1])
        logits = self.classifier(pooled_out)
        
        return logits

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model = BertForAffiliationNameNormalization(2)
model = nn.DataParallel(model, device_ids=[3, 1, 2])
# model = torch.load('checkpoint3/After_epoch_19_bert.pkl')
#model.to(device)
model.train()


optim = AdamW(model.parameters(), lr=5e-5)
loss_func = nn.CrossEntropyLoss()

for epoch in range(20):
    total_train_loss=0
    for i,batch in tqdm(enumerate(train_loader)):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        print('input_id s', input_ids.device)
        attention_mask = batch['attention_mask'].to(device)
        print('attention_mask ', attention_mask.device)
        labels = batch['labels'].to(device)
        print('labels ', labels.device)
        logits = model(input_ids, attention_mask=attention_mask) 
        loss = loss_func(logits.view(-1, 2), labels.view(-1))
        loss.backward()
        optim.step()
        if i%100 == 0:
            print("Epoch: ", epoch, " , step: ", i)
            print("training loss: ", loss.item())
        
        
        total_train_loss+=loss.item()
            
        #if i==1000:
            #torch.save(model, 'checkpoint3/epoch_' + str(epoch) + '_step_' + str(i) + '_bert.pkl')  
    torch.save(model, 'checkpoint7/After_epoch_' + str(epoch) + '_bert.pkl')  
    
    avg_train_loss = total_train_loss / len(train_loader)
    print("avg_train_loss: ", avg_train_loss)

    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
            
        logits = model(input_ids, attention_mask=attention_mask)
        loss = loss_func(logits.view(-1, 2), labels.view(-1))    
        
        total_eval_loss += loss.item()
            
    avg_val_loss = total_eval_loss / len(val_loader)
        
    print("avg_val_loss: ", avg_val_loss)