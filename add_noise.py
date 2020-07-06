import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import uproot
from dataset.py import *
from dp_model import dp_cnn
from loss.py import *
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


'''
# No idea what are these...
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()
'''

# parse arguments
parser = argparse.ArgumentParser(description="dp_cnn")
parser.add_argument("training_path", nargs="?", type=str, default="./data/training", help='path of .root data set to be used for training')
parser.add_argument("validation_path", nargs="?", type=str, default="./data/validation", help='path of .root data set to be used for validation')
parser.add_argument("--num_of_layers", type=int, default=9, help="Number of total layers")
parser.add_argument("--sigma", type=float, default=25, help='noise level')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

# initialization using Xavier methods
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    writer = SummaryWriter(opt.outf)
    
    # Build model
    net = dp_cnn()
    net.apply(init_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    '''
    to be done: loss func?
    '''
    criterion = nn.MSELoss(size_average=False)
    
    
    # training
    writer = SummaryWriter(opt.outf)
    
    

if __name__ == "__main__":
    main()




