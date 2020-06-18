from game2048.module import high_net
from torch.utils.data import DataLoader,dataset,TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''此程序用于云端GPU运行，数据也均在云端存储'''

batch_size = 128
NUM_EPOCHS = 10

#loading data
csv_data_a = pd.read_csv('game2048/data.csv')
csv_data_all = csv_data_a.values
csv_data_h = pd.read_csv('game2048/data2.csv')
csv_data_hh = csv_data_h.values
csv_data_l = pd.read_csv('game2048/data1.csv')
csv_data_ll = csv_data_l.values
csv_data_b = pd.read_csv('game2048/data3.csv')
csv_data_bll = csv_data_b.values

del csv_data_a
del csv_data_h
del csv_data_l
del csv_data_b

board_data_a = csv_data_all[:,0:16]
X_a = np.int64(board_data_a)
board_data_h = csv_data_hh[:,0:16]
X_h = np.int64(board_data_h)
board_data_l = csv_data_ll[0:200000,0:16]
X_l = np.int64(board_data_l)
board_data_b = csv_data_bll[:,0:16]
X_b = np.int64(board_data_b)
X_pr0 = np.concatenate((X_h,X_l,X_a,X_b),axis=0)
X_pr = np.reshape(X_pr0, (-1,4,4))

del board_data_a
del board_data_h
del board_data_l
del board_data_b
del X_a
del X_h
del X_l
del X_b
del X_pr0

direction_data_a = csv_data_all[:,16]
Y_a = np.int64(direction_data_a)
direction_data_h = csv_data_hh[:,16]
Y_h = np.int64(direction_data_h)
direction_data_l = csv_data_ll[0:200000,16]
Y_l = np.int64(direction_data_l)
direction_data_b = csv_data_bll[:,16]
Y_b = np.int64(direction_data_b)
Y_pr = np.concatenate((Y_h,Y_l,Y_a,Y_b),axis=0)

del direction_data_a
del direction_data_h
del direction_data_l
del direction_data_b
del Y_a
del Y_h
del Y_l
del Y_b

del csv_data_all
del csv_data_hh
del csv_data_ll
del csv_data_bll

X, X_test, Y, Y_test = train_test_split(X_pr, Y_pr, test_size=0.001,shuffle=False)

del X_pr
del Y_pr
del X_test
del Y_test

X[X == 0] = 1
X = np.log2(X)

X_train = F.one_hot(torch.LongTensor(X)).permute(0,3,1,2).float()
Y_train = torch.LongTensor(Y)

del X
del Y


train_dataset = TensorDataset(X_train,Y_train)

del X_train
del Y_train

train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True
)


model_high = high_net()
model_high = model_high.cuda()
optimizer = optim.Adam(model_high.parameters(), lr = 0.002)

epoch = 0
for epoch in range(NUM_EPOCHS):
    for data in train_loader:
        img0, label0 = data
        img = Variable(img0).cuda()
        label = Variable(label0).cuda()
        optimizer.zero_grad()
        out = model_high(img)
        loss = F.cross_entropy(out, label)

        loss.backward()
        optimizer.step()

        del img0
        del label0
        del img
        del label
        del out
        del loss

    torch.save(model_high.state_dict(),'game2048/para/high/epoch_{}.pkl'.format(epoch))



