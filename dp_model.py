'''
Adapted with modification from:
https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py
'''

import numpy as np
import torch.nn as nn
from torch.autograd import Variable

#Direct Prediction COnvolutional Network
class dp_cnn(nn.Module):
    def __init__(self, channels=1, num_of_layers=9):
        super(dp_cnn, self).__init__()
        kernel_size = 21
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

'''
class MyLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)
    
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
'''

if __name__=="__main__":
    net = dp_cnn()
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
