import numpy as np
import torch
from torch.autograd import Variable

# calculate the loss of an entire matrix using absolute-loss implementation
def abs_loss(output, target):
    total_loss = np.abs(output-target)
    loss_sum = np.sum(total_loss)
    return loss_sum / output.size

# We use the absolute value loss function
# Calculate the average loss of patches devided
def patch_abs_loss(output, target):

    # patch size TBD
    patch_size = 5
    
    #Slice the tensor into patches with patch_size
    patch_out = output.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patch_tgt = target.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    
    #Calculate the abs_loss of each patches
    num_patch = patch_tgt.nelement()
    sum_loss = 0
    
    #sum-up the absolute loss of all the patches
    for i in range(list(patch_tgt.size())[1]):
        sum_loss += abs_loss(patch_out[0][i],patch_tgt[0][i])
    
    #return the average of the absolute loss of each patches
    return sum_loss/num_patch

# Sample Script
if __name__=="__main__":
    dtype = torch.FloatTensor
    N, D_in, D_out = 64, 1000,  10
    x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
    y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)
    direst_loss = abs_loss(x,y)
    patch_loss = patch_abs_loss(x,y)
    print("Direct-calculated absolute loss is",direst_loss)
    print("patch-based absolute loss is",patch_loss)
