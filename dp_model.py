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
'''

class PatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        for i in range(len(output)):
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            max_patch_loss = 0
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    max_patch_loss = max(max_patch_loss, f.l1_loss(output_patches[i][j], target_patches[i][j]))
            avg_loss+=max_patch_loss
        avg_loss/=(list(output_patches.size())[0] * (list(output_patches.size())[1]))
        return avg_loss;

class WeightedPatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            devider = 0
            weighted_loss = 0
            # calculate loss for each patch of the image
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    weighted_loss += f.l1_loss(output_patches[i][j],target_patches[i][j]) * torch.mean(target_patches[i][j])
                    devider += torch.mean(target_patches[i][j])
            avg_loss += weighted_loss/devider
        avg_loss/=(list(output_patches.size())[0] * (list(output_patches.size())[1]))
        return avg_loss;

class FilteredPatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size, filter_rate):
        avg_loss = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            
            valid_loss = 0
            invalid_count = 0
            # calculate loss for each patch of the image
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    if torch.mean(target_patches[i][j]) <= filter_rate:
                        invalid_count += 1
                    else:
                        valid_loss += f.l1_loss(output_patches[i][j],target_patches[i][j])
            avg_loss += valid_loss
            devider = (list(output_patches.size())[0]) * (list(output_patches.size())[1])
            devider -= invalid_count
        return avg_loss/devider

if __name__=="__main__":
    net = dp_cnn()
    input = torch.randn(1, 1, 128, 128)
    out = net(input)
    print(out)
