import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import uproot
from dataset import *
from dp_model import dp_cnn
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = argparse.ArgumentParser(description="dp_cnn")
parser.add_argument("training_path", nargs="?", type=str, default="./data/training", help='path of .root data set to be used for training')
parser.add_argument("validation_path", nargs="?", type=str, default="./data/validation", help='path of .root data set to be used for validation')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--num_of_layers", type=int, default=9, help="Number of total layers")
parser.add_argument("--sigma", type=float, default=25, help='noise level')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():
    #check if it is cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("We are currently using" + device + "for our training...\n")
    
    # Build model
    net = dp_cnn()
    net.apply(init_weights)
    
    #I'll be using a built-in loss function to guarantee that it is working
    criterion = nn.L1Loss()
    
    # Move to GPUs if available
    if device == "cuda:0":
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        criterion.cuda()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    for epoch in range(opt.epochs):
        training_files = glob.glob(os.path.join(opt.training_path, '*.root'))
        for training_file in training_files:
            branch = get_all_histograms(training_file)
            length = np.size(branch)
            for i in range(length):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                # prepare data
                data = get_bin_weights(branch,0).copy()
                noisy = add_noise(data, args.sigma).copy()
                data = torch.from_numpy(data).to(device)
                noisy = torch.from_numpy(noisy)
                noisy = noisy.unsqueeze(0).unsqueeze(1).to(device)
                out_train = model(noisy.float()).to(device)
                loss = criterion(out_train.squeeze(0).squeeze(0), data)
                loss.backward()
                optimizer.step()
                model.eval()
                if step % 10 ==0:
                    train_loss = np.append(train_loss, loss.item())
                step += 1
            model.eval()
        # validation
        validation_files = glob.glob(os.path.join(opt.validation_path,'*root'))
        total_loss = 0
        count = 0
        for validation_file in validation_files:
            branch = get_all_histograms(training_file)
            length = np.size(branch)
            for i in range(length):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                # prepare data
                data = get_bin_weights(branch,0).copy()
                noisy = add_noise(data, args.sigma).copy()
                data = torch.from_numpy(data).to(device)
                noisy = torch.from_numpy(noisy)
                noisy = noisy.unsqueeze(0).unsqueeze(1).to(device)
                out_train = model(noisy.float()).to(device)
                total_loss += criterion(out_train.squeeze(0).squeeze(0), data)
            avg_loss = total_loss / length
            writer.add_scalar('Loss values on validation data', avg_loss, count)
            count += 1
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    main()
