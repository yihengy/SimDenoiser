'''
Adapted with modification from:
https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py
'''

import numpy as np
import torch.nn as nn

#Direct Prediction COnvolutional Network
class dp_cnn(nn.Module):

    def __init__(self, num_chanels=1, num_layers=9):
        super(dp_cnn,self).__init__()
        
        #ker_size, pad_size, features, and bias_val should be determined by experiment
        ker_size = 21 #TBD
        pad_size = 1 #to avoid edge-data being underrepresented
        features = 32 #tbd
        '''
        **********Network structure**********
        In each layer l, the network applies a linear convolution
        to the output of the previous layer, adds a constant bias,
        and then applies an element-wise nonlinear transformation
        (in this case ReLU)
        '''
        layers = []
        
        layers.append(nn.Conv2d(in_channels = channels,out_channels = features,kernel_size = ker_size,padding=pad_size, bias = True))
        layers.append(nn.ReLU(inplace=True))
        
        for t in range num_layers-2
            layers.append(nn.Conv2d(in_channels = channels,out_channels = features,kernel_size = ker_size,padding=pad_size, bias = True))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Conv2d(in_channels = features,out_channels = channels,kernel_size = ker_size,padding=pad_size, bias = True))
        self.dp_cnn = nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.dp_cnn(x)
        return x
    
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
if __name__=="__main__":
    net = dp_cnn()
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
