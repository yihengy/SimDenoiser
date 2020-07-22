"""
Adapted from https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=9):
        super(DnCNN, self).__init__()
        kernel_size = 5
        padding = 2
        features = 100
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

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
'''
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
'''

class WeightedPatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            weighted_loss = 0
            # calculate loss for each patch of the image
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    weighted_loss += f.l1_loss(output_patches[i][j],target_patches[i][j]) * torch.mean(target_patches[i][j])
            avg_loss+=weighted_loss
        avg_loss/=(list(output_patches.size())[0] * (list(output_patches.size())[1]))
        return avg_loss;

class FilteredPatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size, filter_rate):
        loss = 0
        devider = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            
            # calculate loss for each patch of the image
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    print(torch.mean(target_patches[i][j]), filter_rate)
                    input()
                    if torch.mean(target_patches[i][j]) > filter_rate:
                        loss += f.l1_loss(output_patches[i][j],target_patches[i][j])
                        devider += 1
        return loss/devider


if __name__=="__main__":
    criterion_1 = PatchLoss()
    criterion_2 = WeightedPatchLoss()
    criterion_3 = FilteredPatchLoss()
    dtype = torch.FloatTensor
    x = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    y = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    loss_1 = criterion_1(x, y, 10)
    loss_2 = criterion_2(x, y, 10)
    loss_3 = criterion_3(x, y, 10, 0.1)
    print("Test loss: ")
    print(str(loss_1))
    print(str(loss_2))
    print(str(loss_3))
    net = DnCNN()
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
