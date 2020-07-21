import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from dataset import *
import glob
import uproot
from torch.utils.data import DataLoader
import numpy as np
from models import DnCNN
import torch.optim as optim

class PatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        num_true=0
        num_false = 0
        for i in range(len(output)):
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            losses = np.zeros((list(output_patches.size())[0] * (list(output_patches.size())[1])))
            means = np.zeros((list(output_patches.size())[0] * (list(output_patches.size())[1])))
            count = 0
            # calculate loss for each patch of the image
            for i in range(list(output_patches.size())[0]):
                for j in range(list(output_patches.size())[1]):
                    losses[count] = f.l1_loss(output_patches[i][j], target_patches[i][j])
                    means[count] = torch.mean(target_patches[i][j])
                    count+=1
            if (np.argmax(losses) == np.argmax(means)):
                num_true+=1
            else:
                num_false+=1
        total_num = num_true + num_false
        correctness = num_true/ total_num
        return total_num, num_true, num_false, correctness

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_uniform_(m.weight)

if __name__ == "__main__":
    dataset = RootDataset(root_file='test.root', sigma = 15)
    loader = DataLoader(dataset=dataset, batch_size=100)
    model = DnCNN(channels = 1, num_of_layers=9)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    criterion = PatchLoss()
    for i, data in enumerate(loader, 0):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        truth, noise = data
        
        output = model(noise.unsqueeze(1).float())
        total_num, num_true, num_false, correctness= criterion(output.squeeze(1), truth, 50)
        print("Total:   " + str(total_num))
        print("Correct:   " + str(num_true))
        print("Incorrect: " + str(num_false))
        print("Correctness: " + str(correctness * 100) + "%")
        

