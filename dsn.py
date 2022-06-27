import torch
import torch.nn as nn
import torch.nn.functional as F

Fs = 100

class DeepSleepNet(nn.Module):

    def __init__(self, ch=1):
        
        super(DeepSleepNet, self).__init__()
        
        # Do I need the bias?
        # How does kernel size and stride affect results?
        # How does activation affect results?
        # Should we try multiple conv layers for different timescales?
        
        # linear kernels, nonlinear readout
        # self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(7,50),stride=10,bias=False)        
        # self.act = nn.Softmax()
        self.read = nn.Linear(9344, 5)
        
        # linear kernels, nonlinear node/activation, linear readout
        # linear kernels, nonlinear reservoir, linear readout

    def forward(self, x):

        x = torch.flatten(x, 0, 1)
        x = torch.flatten(x, 1, 2)
        x = self.read(x)
        print(x.shape)
        y = x.view(10,25,5)
        y = torch.transpose(y,1,2)
        print(y.shape)
        return y

    # def bias_value(self):
        # return self.conv1.bias.data, self.conv2.bias.data
