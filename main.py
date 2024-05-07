# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:17:07 2021

@author: Administrator
"""

from torch import nn
from torch.nn import functional as F
import torch
import json
import os
import numpy as np
import pandas as pd
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Dataset

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

train_file = Path('82train1.json')  #/mnt/xfs_code/8_channel_code/train_2.json
train_set = SimpleDateset(train_file,transforms)
train_loader = DataLoader(train_set,batch_size=8,shuffle=True,num_workers=2)

test_file = Path('82test1.json')
test_set = SimpleDateset(test_file, transforms)
test_loader  = DataLoader(test_set,batch_size=8,shuffle=False,num_workers=2)
        
class Net(nn.Module):
    def __init__(self,shortcut=None,num_classes=6):
        super(Net,self).__init__()
        # DEM
        self.conv1_d = nn.Conv2d(1, 32, kernel_size=7,stride=3,padding=2)  # 5*5卷积块
        self.conv2_d = nn.Conv2d(32, 32, kernel_size=5, stride=2,padding=2)
        self.conv3_d = nn.Conv2d(32, 32, kernel_size=3, stride=2,padding=1)
        self.conv4_d = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.BatchNorm_d = nn.BatchNorm2d(32)
        # 遥感
        self.conv1_y = nn.Conv2d(4, 32, kernel_size=(7,7), stride=3)
        self.conv2_y = nn.Conv2d(32, 32, kernel_size=(5,5), stride=2,padding=1)
        self.conv3_y = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2,padding=1)
        self.conv4_y = nn.Conv2d(32, 64, kernel_size=(3,3), stride=2,padding=1)
#         self.BatchNorm_y = nn.BatchNorm3d(32)
        self.BatchNorm_y = nn.BatchNorm2d(32)
        # 岩性
        self.conv1_r = nn.Conv2d(1, 32, kernel_size=7, stride=3)
        self.conv2_r = nn.Conv2d(32, 32, kernel_size=5, stride=2,padding=1)
        self.conv3_r = nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1)
#         self.conv4_r = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.BatchNorm_r = nn.BatchNorm2d(32)
        # 土壤
        self.conv1_s = nn.Conv2d(1, 32, kernel_size=7, stride=3)
        self.conv2_s = nn.Conv2d(32, 32, kernel_size=5, stride=2,padding=1)
        self.conv3_s = nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1)
#         self.conv4_s = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.BatchNorm_s = nn.BatchNorm2d(32)
        # 植被
        self.conv1_p = nn.Conv2d(1, 32, kernel_size=7, stride=3)
        self.conv2_p = nn.Conv2d(32, 32, kernel_size=5, stride=2,padding=1)
        self.conv3_p = nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1)
#         self.conv4_p = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.BatchNorm_p = nn.BatchNorm2d(32)
        
#         self.conv7 = DepthSeperabelConv2d(320, 64, kernel_size=3, stride=2)

        self.avg_pool = nn.AvgPool2d(2,2)
        self.max_pool = nn.MaxPool2d(2,2)
        self.activation1 = nn.ReLU()
        
        self.fc1 = nn.Linear(in_features=194688, out_features=512, bias=True)  # 216320,9216,194688,173056,129792,8x194688 and 237952x512
        self.fc2 = nn.Linear(in_features=512, out_features=6, bias=True)
        self.fc3 = nn.Linear(in_features=512, out_features=2, bias=True)

        self.Res2 = nn.MaxPool2d(2,2)
        self.Res1 = nn.MaxPool2d(2,2)
        self.Res3 = nn.MaxPool2d(4,4)

    def forward(self,x):
        input_size = x.size(0) #获取batch_size   
        # 切分数据
        dem  = x[:, :, 0, :, :]    # [8, 1, 1280, 1280]
        yg  = x[:, :, 1:5, :, :]  # [8, 1, 4, 1280, 1280]
        rock = x[:, :, 5, :, :]    # [8, 1, 1280, 1280]
        soil = x[:, :, 6, :, :]    # [8, 1, 1280, 1280]
        plant = x[:, :, 7, :, :]   # [8, 1, 1280, 1280]
#         hyper = hyper.squeeze(1)      
        # DEM
        x1 = self.conv1_d(dem)  # [8, 32, 426, 426]
        x1 = self.BatchNorm_d(x1)
        x1 = self.activation1(x1)  
        res1 = x1
        res1 = self.Res1(res1)
        x1 = self.conv2_d(x1)  # [8, 32, 212, 212]
        x1 = self.activation1(x1)
        x1 = x1 + res1
        x1 = self.conv3_d(x1)  # [8, 32, 106, 106]
        res1 = x1
        res1 = self.Res1(res1)
        x1 = self.conv4_d(x1)  # [8, 64, 54, 54]
        x1 = self.activation1(x1) 
        x1 = x1 + res1
        # 遥感
        yg = yg.squeeze(1)
        x2 = self.conv1_y(yg)  # [8, 32, 2, 425, 425]
        x2 = self.BatchNorm_y(x2)
        x2 = self.activation1(x2)
        res2 = x2
        res2 = self.Res3(res2)
        x2 = self.conv2_y(x2)  # [8, 32, 2, 212, 212]
        x2 = self.activation1(x2)
        x2 = self.conv3_y(x2)  # [8, 32, 2, 106, 106]
        x2 = x2 + res2
        x2 = self.conv4_y(x2)  # [8, 64, 1, 54, 54]
        x2 = self.activation1(x2) 
#         x2 = x2.reshape(x2.size(0),x2.size(1)*x2.size(2),x2.size(3),x2.size(4)) # [8, 64, 52, 52]
        # 岩性
        x3 = self.conv1_r(rock)  # [8, 32, 425, 425]
        x3 = self.BatchNorm_r(x3) 
        x3 = self.activation1(x3)  
        res1 = x3
        res1 = self.Res1(res1)
        x3 = self.conv2_r(x3)  # [8, 32, 212, 212]
        x3 = self.activation1(x3)
        x3 = x3 + res1
        x3 = self.conv3_r(x3)  # [8, 32, 106, 106]
        x3 = self.max_pool(x3)  # [8, 32, 53, 53]
#         x3 = self.conv4_r(x3)  # [8, 64, 52, 52]
        x3 = self.activation1(x3) 
        # 土壤
        x4 = self.conv1_s(soil)
        x4 = self.BatchNorm_s(x4)
        x4 = self.activation1(x4)  
        res1 = x4
        res1 = self.Res1(res1)
        x4 = self.conv2_s(x4) 
        x4 = self.activation1(x4)
        x4 = x4 + res1
        x4 = self.conv3_s(x4)
        x4 = self.max_pool(x4)
#         x4 = self.conv4_s(x4)
        x4 = self.activation1(x4) 
        # 植被
        x5 = self.conv1_p(plant)
        x5 = self.BatchNorm_p(x5)
        x5 = self.activation1(x5)  
        res1 = x5
        res1 = self.Res1(res1)
        x5 = self.conv2_p(x5) 
        x5 = self.activation1(x5)
        x5 = x5 + res1      
        x5 = self.conv3_p(x5)
        x5 = self.max_pool(x5)
#         x5 = self.conv4_p(x5)
        x5 = self.activation1(x5) 
        # 数据融合
        x = torch.cat((x1,x2,x3,x4,x5),1)
#         x = self.conv7(x)
        x = self.activation1(x)
        x = self.max_pool(x)
        x = self.activation1(x)
        x = x.view(input_size,-1) #拉平
        x = self.fc1(x)
        output1 = self.fc2(x)
        output2 = self.fc3(x)

        return output1,output2

# 定义损失函数
class My_loss(nn.Module):
    def __init__(self, n_classes = 6):
        super(My_loss, self).__init__()
    def forward(self, output1, labels):    # labels表示真实标签
        p = F.softmax(output1, dim = 1)
        loss = self.cal_loss(labels, output1)
        return loss
    def cal_loss(self, labels, output1):
        loss = 0
        for i in range(output1.shape[0]):
            if labels[i] == 0:
                loss += -0.05*(167/9)*(torch.log(F.softmax(output1[i])[0]))
#                 print(F.softmax(output1[i])[0])
            elif labels[i] == 1:
                loss += -0.05*(167/51)*( torch.log(F.softmax(output1[i])[1]))
            elif labels[i] == 2:
                loss += -0.05*(167/22)*( torch.log(F.softmax(output1[i])[2]))
            elif labels[i] == 3:
                loss += -0.05*(167/8)*(torch.log(F.softmax(output1[i])[3]))
            elif labels[i] == 4:
                loss += -0.05*(167/23)*( torch.log(F.softmax(output1[i])[4]))
            elif labels[i] == 5:
                loss += -0.05*(167/54)*(torch.log(F.softmax(output1[i])[5]))
        return loss
    
net=Net().to(device)

# criterion=nn.CrossEntropyLoss()
criterion=My_loss()
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
        output1,output2=net(images)
        train_loss+=criterion(output1,labels).item()
        loss=criterion(output1,labels)
        loss.backward()
        optimizer.step()
        pred = output1.data.max(1,keepdim=True)[1]
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
            output1,output2=net(data)
            output1=F.softmax(output1)
            output2=F.softmax(output2)
#             print(output1)
#             print(output2)
            res1.append(output1)
            res2.append(output2)
            test_loss+=criterion(output1,label).item()
            predict = output1.data.max(1,keepdim=True)[1]
            print('predict:',predict.tolist())
            
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
        torch.save(net.state_dict(), '82net1.pkl')  # 只保存模型参数
        print('save model') 
