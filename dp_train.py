import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
import argparse
from models import dp_cnn
from dataset import *
import glob
import torch.optim as optim
import uproot
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = argparse.ArgumentParser(description="dp_cnn")
parser.add_argument("training_path", nargs="?", type=str, default="./data/training", help='path of .root data set to be used for training')
parser.add_argument("validation_path", nargs="?", type=str, default="./data/validation", help='path of .root data set to be used for validation')
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dp_cnn(channels=1, num_of_layers=opt.num_of_layers).to(device)
    model.apply(init_weights)
    criterion = nn.MSELoss(size_average=False)
    criterion.to(device=opt.device)
    optimizer = optim.Adam(model.parameters(), lr = opt.lr)
    loss_per_epoch = np.zeros(opt.epochs)
    
    # train the net
    step = 0
    for epoch in range(opt.epochs):
        print("Beginning epoch " + str(epoch))
        training_files = glob.glob(os.path.join(opt.training_path, '*.root'))
        for training_file in training_files:
            print("Opened file " + training_file)
            branch = get_all_histograms(training_file)
            length = np.size(branch)
            for i in range(length):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                data = get_bin_weights(branch, 0).copy()
                noisy = add_noise(data, opt.sigma).copy()
                data = torch.from_numpy(data).to(device)
                noisy = torch.from_numpy(noisy)
                noisy = noisy.unsqueeze(0)
                noisy = noisy.unsqueeze(1).to(device)
                out_train = model(noisy.float()).to(device)
                loss = criterion(out_train.squeeze(0).squeeze(0), data)
                loss.backward()
                optimizer.step()
                model.eval()
            model.eval()

        # validation
        validation_files = glob.glob(os.path.join(opt.validation_path, '*root'))
        # peak signal to noise ratio
        epoch_loss = 0
        count = 0
        for validation_file in validation_files:
            print("File: " + validation_file)
            branch = get_all_histograms(validation_file)
            length = np.size(branch)
            for i in range (length):
                # get data (ground truth)
                data = get_bin_weights(branch, 0).copy()
                # add noise
                noisy = add_noise(data, opt.sigma).copy()
                # convert to tensor
                data = torch.from_numpy(data).to(device)
                noisy = torch.from_numpy(noisy)
                noisy = noisy.unsqueeze(0)
                noisy = noisy.unsqueeze(1).to(device)
                out_train = model(noisy.float()).to(device)
                loss = criterion(out_train.squeeze(0).squeeze(0), data)
                epoch_loss+=loss.item()
            epoch_loss/=length
            count+=1
        # save the model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
        loss_per_epoch[epoch] = epoch_loss
        print("Average loss per image in epoch " + str(epoch) + " of " + str(opt.epochs-1) +": "+ str(epoch_loss))
    loss_plot = plt.plot(loss_per_epoch)
    plt.savefig("loss_plot.png")
    
if __name__ == "__main__":
    main()
