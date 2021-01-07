import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from numpy.linalg import norm

def create_dataset(datas, look_back):
    dataset = datas.astype('float32')
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x: x / scalar, dataset))
    #look_back = 3
    #window_wid = 2*look_back

    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return dataset, np.array(dataX), np.array(dataY)

class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=3):
        super(lstm_reg, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)#,dropout=1)
          #bidirectional=True) # rnn
        self.reg = nn.Linear(hidden_size, output_size) # regression
        
    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h)
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

def train(datas,look_back=5,step = 200,epochtime = 100):
    dataset, data_X, data_Y = create_dataset(datas,look_back)
    train_size = int(len(data_X)*0.9)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]
    train_X = train_X.reshape(-1, 1, look_back)
    train_Y = train_Y.reshape(-1, 1, 1)
    test_X = test_X.reshape(-1, 1, look_back)

    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)
    test_x = torch.from_numpy(test_X)
    
    net = lstm_reg(look_back, 2*look_back)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
    loss_list = []
    for e in range(epochtime):
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        out = net(var_x)
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if (e + 1) % 10 == 0:
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
    
    net = net.eval() 
    data_X = data_X.reshape(-1, 1, look_back)
    data_X = torch.from_numpy(data_X)
    var_data = Variable(data_X)
    pred_test = net(var_data)
    pred_test = pred_test.view(-1).data.numpy()
    pred_test = np.concatenate((np.array(dataset)[:look_back],pred_test))
    
    return pred_test,loss_list