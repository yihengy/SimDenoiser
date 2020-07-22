import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
import numpy as np

'''
class PatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        print("length:")
        print(len(output))
        for i in range(len(output)):
            output_patches = output[i].unfold(0,1,1).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            target_patches = target[i].unfold(0,1,1).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            max_patch_loss = 0
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    print("Patch Tag: i = "+str(i)+", j = "+str(j)+":")
                    print("Output Patch:")
                    print(output_patches[i][j])
                    print("Target Patch:")
                    print(target_patches[i][j])
                    tmp = f.l1_loss(output_patches[i][j],target_patches[i][j])
                    print("L1loss de Patch:"+str(tmp))
                    print("Max before this patch: " + str(max_patch_loss))
                    max_patch_loss = max(max_patch_loss, tmp)
                    print("Max after this patch: " + str(max_patch_loss))
            avg_loss+=max_patch_loss
        avg_loss/=(list(output_patches.size())[0] * (list(output_patches.size())[1]))
        return avg_loss;
'''

class PatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        output_patches = output.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        target_patches = target.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        max_patch_loss = 0
        for i in range(list(output_patches.size())[0]):
            for j in range(list(output_patches.size())[1]):
                print("Patch Tag: i = "+str(i)+", j = "+str(j)+":")
                print("Output Patch:")
                print(output_patches[i][j])
                print("Target Patch:")
                print(target_patches[i][j])
                tmp = f.l1_loss(output_patches[i][j],target_patches[i][j])
                print("L1loss de Patch:"+str(tmp))
                print("Max before this patch: " + str(max_patch_loss))
                max_patch_loss = max(max_patch_loss, tmp)
                print("Max after this patch: " + str(max_patch_loss))
        return max_patch_loss

if __name__ == "__main__":
    criterion = PatchLoss()
    dtype = torch.FloatTensor
    # Ordinary Check
    x = Variable(torch.randn(5, 5).type(dtype), requires_grad=False)
    print(x)
    y = Variable(torch.randn(5, 5).type(dtype), requires_grad=False)
    print(y)
    loss = criterion(x,y,3)
    print(loss)
    print("Finish testing.")
    
