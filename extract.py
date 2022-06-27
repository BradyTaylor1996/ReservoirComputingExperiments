import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):

    def __init__(self, ch=1):
        
        super(Extractor, self).__init__()
        
        # Do I need the bias?
        # How does kernel size and stride affect results?
        # How does activation affect results?
        # Should we try multiple conv layers for different timescales?
        
        # linear kernels, nonlinear readout
        self.conv1 = nn.Conv1d(in_channels=7,out_channels=64,kernel_size=100,stride=20,bias=False)
        # self.conv2 = nn.Conv1d(in_channels=64,out_channels=32,kernel_size=400,stride=100,bias=False)
        # linear kernels, nonlinear node/activation, linear readout
        # linear kernels, nonlinear reservoir, linear readout

    def forward(self, x):
    
        x = self.conv1(x)
        
        return x

