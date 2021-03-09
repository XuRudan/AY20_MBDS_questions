# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:26:38 2021

@author: ASUS
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
   
class DatasetFromArray(Dataset):
    def __init__(self, dataarray, labelarray):
        self.dataarray = dataarray
        self.labelarray = labelarray

    def __getitem__(self, index):
        data = torch.tensor(self.dataarray[index,:]).reshape(1,-1)
        label = torch.tensor(self.labelarray[index,:]).squeeze()
        sample = {'data': data, 'label': label}
        return sample

    def __len__(self):
        return len(self.labelarray)

class MLP(nn.Module):
    def __init__(self,insize,hiddensize,outsize):
        super().__init__()
        self.L1 = nn.Linear(insize,hiddensize)
        self.L2 = nn.Linear(hiddensize,hiddensize)
        self.L3 = nn.Linear(hiddensize,outsize)
        
    def forward(self,x):
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        return x.squeeze()
    
def train(model, criterion, optimizer, dataloader, epochs=1, dataset_size=10000):
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch,epochs-1))
        print('-'*10)
        model.train()
        running_loss = 0.0
        for data in dataloader:
            inputs, labels = data['data'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            #前向传播 计算损失
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            #训练阶段反向优化
            loss.backward()
            optimizer.step()
            #统计
            running_loss += loss.data
        epoch_loss = running_loss / dataset_size
        print('{} Loss: {:.4f}'.format('train',epoch_loss))
        print()
    return model

train_data_path = 'F:/AY20_MBDS_questions-answer/Question 3/train_data.txt'
train_label_path = 'F:/AY20_MBDS_questions-answer/Question 3/train_truth.txt'
test_data_path = 'F:/AY20_MBDS_questions-answer/Question 3/test_data.txt'
outfilepath = 'F:/AY20_MBDS_questions-answer/Question 3/test_predicted.txt'
batch_size = 32
epochs = 40
learning_rate = 0.001

traindata = pd.read_csv(train_data_path, sep = '\t', header = 'infer').values.astype(np.float32)
trainlabel = pd.read_csv(train_label_path, sep = '\t', header = 'infer').values.astype(np.float32)
traindataset = DatasetFromArray(traindata, trainlabel)
traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

net = MLP(3,4,1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

model = train(net, criterion, optimizer, traindataloader, epochs=epochs)
testdata = pd.read_csv(test_data_path, sep = '\t', header = 'infer').values.astype(np.float32)
testdata = torch.tensor(testdata)
test_predicted = model(testdata)
output = pd.DataFrame({'y':test_predicted.detach().numpy()})
output.to_csv(path_or_buf = outfilepath, index=False)




