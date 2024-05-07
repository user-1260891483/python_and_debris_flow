# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:17:07 2021

@author: Administrator
"""

from torch import nn
from torch.nn import functional as F
import torch
import json
import numpy as np
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Dataset
from ResNet import resnet18,resnet34
from SENet import senet
from VGG16 import VGGNet
from EfficientNet import EfficientNet
from EfficientNet import ShuffleNetV1
from EfficientNet import MobileNet


class SimpleDateset(Dataset):
    def __init__(self,data_file,transform):
        with open(data_file,'r') as f:
            self.meta = json.load(f)
        self.transform_d = transform
    def __getitem__(self,i):
        data_path = Path(self.meta['data_path'][i])
        data = np.load(data_path,allow_pickle=True)
        data = self.transform_d(data)        
        data = data.permute(1,2,0)
        data = data.unsqueeze(0)     
        label = int(self.meta['data_labels'][i])
        return data,label
    def __len__(self):
        return len(self.meta['data_labels'])
       
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100

transforms = transforms.Compose([transforms.ToTensor()])

train_file = Path('82train1.json')
train_set = SimpleDateset(train_file,transforms)
train_loader = DataLoader(train_set,batch_size=8,shuffle=True,num_workers=2)

test_file = Path('82test1.json')
test_set = SimpleDateset(test_file, transforms)
test_loader  = DataLoader(test_set,batch_size=8,shuffle=False,num_workers=2)


# net=Net().to(device)
# net=Ensnet(n_labels=6).to(device)
# net = create_regnet(model_name='RegNetY_400MF',num_classes=6).to(device) 
# net = create_model(num_classes=6).to(device)
net=resnet18().to(device)
# net=VGGNet().to(device)
# net=MobileNet().to(device)
# net=ShuffleNetV1(group=3).to(device)
# net=EfficientNet(width_coefficient=1.0,depth_coefficient=1.0,dropout_rate=0.2,num_classes=6).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)

train_acc_list = []
train_ls = []
test_acc_list = []
test_two_list = []
res1 = []
res2 = []
res3 = []
best_acc6 = 0
best_acc2 = 0
min_epoch = 20
#训练与调试
for epoch in range(1,epochs+1):
    net.train()
    train_loss=0
    train_correct=0
    for i,(images,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images,labels=images.float().to(device),labels.to(device)
        output=net(images)
        train_loss+=criterion(output,labels).item()
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1,keepdim=True)[1]
        train_correct+=pred.eq(labels.data.view_as(pred)).cpu().sum()
    train_loss/=len(train_loader.dataset)
    
    train_acc = 100.0 * train_correct / len(train_loader.dataset)
    train_acc= float(train_acc)
    train_acc_list.append(train_acc)
    train_ls.append(train_loss)

    print('Train{} \nAverage loss:{:.5f} \nAccuracy:{:3f}%\n'.format(epoch,train_loss,100.0*train_correct/len(train_loader.dataset)))
#     print('Train Epoch:{}-{} \t Loss:{:.5f}'.format(epoch,i,loss.item()))
    net.eval()
    t2 = 0
    correct_01=0
    test_correct=0
    test_loss=0
    test_correct=0
    
    with torch.no_grad():
        for data,label in test_loader:
            data,label = data.float().to(device),label.to(device)
            output=net(data)
            output=F.softmax(output)
#             print(output)
            res1.append([output])
            test_loss+=criterion(output,label).item()
            predict = output.data.max(1,keepdim=True)[1]
            print('pred:',predict.tolist())
            
            r = label.tolist()
            p = predict.tolist()
            p = [i[0] for i in p]
            for r_,p_ in zip(r, p):
                if(r_<=2 and p_<=2) or (r_>=3 and p_>=3):
                    correct_01 += 1
            t2 = correct_01

            test_correct+=predict.eq(label.data.view_as(predict)).cpu().sum()
        test_loss /= len(test_loader.dataset)

        test_acc = 100.0 * test_correct / len(test_loader.dataset)
        test_acc = float(test_acc)
        test_acc_list.append(test_acc)
        
        test_2 = 100.0 * t2 / len(test_loader.dataset)
        test_2 = float(test_2)
        test_two_list.append(test_2)
    print(f'6分类Test{epoch} \nAverage loss:{test_loss:.5f} \nAverage Accuracy:{100.0*test_correct/len(test_loader.dataset):3f}%\n')
    print(f'2分类Test{epoch} \nAverage loss:{test_loss:.5f} \nAverage Accuracy:{100.0*t2/len(test_loader.dataset):3f}%\n')
    a = 100.0*test_correct/len(test_loader.dataset)
    b = 100.0*t2/len(test_loader.dataset)
    if best_acc6 <= a and best_acc2 <= b and epoch >= min_epoch:
        best_acc6 = a
        best_acc2 = b                      
        torch.save(net.state_dict(), '82res18CE1.pkl')  # 只保存模型参数
        print('save model') 
