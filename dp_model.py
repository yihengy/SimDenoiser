'''
Adapted with modification from:
https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py
'''

import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

#Direct Prediction COnvolutional Network
class dp_cnn(nn.Module):

    def __init__(self, num_chanels=1, num_of_layers=9):
        super(dp_cnn,self).__init__()
        
        #ker_size, pad_size, features, and bias_val should be determined by experiment
        ker_size = 21 #TBD
        pad_size = 1 #to avoid edge-data being underrepresented
        features = 64 #tbd
        '''
        **********Network structure**********
        In each layer l, the network applies a linear convolution
        to the output of the previous layer, adds a constant bias,
        and then applies an element-wise nonlinear transformation
        (in this case ReLU)
        '''
        layers = []
        
        layers.append(nn.Conv2d(in_channels = channels,out_channels = features,kernel_size = ker_size,padding=pad_size, bias = False))
        layers.append(nn.ReLU(inplace=True))
        
        for t in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels = channels,out_channels = features,kernel_size = ker_size,padding=pad_size, bias = False))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Conv2d(in_channels = features,out_channels = channels,kernel_size = ker_size,padding=pad_size, bias = False))
        self.dp_cnn = nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.dp_cnn(x)
        return out
    
    '''
    def num_flat_features(self, x):
        # To be implemented
        
        The value of "features" would be perfect if it is exactly
        the size of the matrix of the image, and the output will
        exactly be the denoised value of each pixel.
        
        We want to implement a feature-calculating function to
        alter the dimension of the final layers (or perhaps we
        can simply use that value for all layers).
        
    '''

class MyLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MyLoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, output, target, patch_size):
        # split output and target images into patches
        #Slice the tensor into patches with patch_size
        patch_out = output.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patch_tgt = target.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        
        #Calculate the abs_loss of each patches
        num_patch = patch_tgt.nelement()
        sum_loss = 0
        
        #sum-up the absolute loss of all the patches
        for i in range(list(patch_tgt.size())[1]):
            sum_loss += f.l1_loss(patch_out[0][i],patch_tgt[0][i])
        
        #return the average of the absolute loss of each patches
        return sum_loss/num_patch
    

if __name__=="__main__":
    criterion = MyLoss()
    dtype = torch.FloatTensor
    x = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    y = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    loss = criterion(x, y, 10)
    print(loss)
    net = dp_cnn()
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
