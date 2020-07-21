# modified from github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from models import DnCNN, PatchLoss
from dataset import *
import glob
import torch.optim as optim
import uproot
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse arguments
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("training_path", nargs="?", type=str, default="./data/training", help='path of .root data set to be used for training')
parser.add_argument("validation_path", nargs="?", type=str, default="./data/validation", help='path of .root data set to be used for validation')
parser.add_argument("--num_of_layers", type=int, default=9, help="Number of total layers")
parser.add_argument("--sigma", type=float, default=10, help='noise level')
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--trainfile", type=str, default="test.root", help='path of .root file for training')
parser.add_argument("--valfile", type=str, default="test.root", help='path of .root file for validation')
parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
parser.add_argument("--model", type=str, default=None, help="Existing model, if applicable")
args = parser.parse_args()

'''
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_uniform_(m.weight)
'''
def init_weights(m):
    if type(m) == nn.Linear:
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)

def main():
    # choose cpu or gpu
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    print('Loading Dataset--')
    dataset_train = RootDataset(root_file=args.trainfile, sigma = args.sigma)
    loader_train = DataLoader(dataset=dataset_train, batch_size=args.batchSize)
    dataset_val = RootDataset(root_file=args.valfile, sigma=args.sigma)
    val_train = DataLoader(dataset=dataset_val)

    # Build model
    model = DnCNN(channels=1, num_of_layers=args.num_of_layers).to(device=args.device)
    if (args.model == None):
        model.apply(init_weights)
        print("Creating new model ")
    else:
        print("Loading model from file " + args.model)
        model.load_state_dict(torch.load(args.model))
        model.eval()

    # Loss function
    criterion = PatchLoss()
    criterion.to(device=args.device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, verbose=True)

    # training and validation
    step = 0
    training_losses = np.zeros(args.epochs)
    validation_losses = np.zeros(args.epochs)
    for epoch in range(args.epochs):
        print("Epoch #" + str(epoch))
        # training
        train_loss = 0
        for i, data in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            truth, noise = data
            noise = noise.unsqueeze(1)
            output = model(noise.float().to(args.device))
            batch_loss = criterion(output.squeeze(1).to(args.device), truth.to(args.device), 10).to(args.device)
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            model.eval()
        training_losses[epoch] = train_loss/len(dataset_train)
        print("Train: "+ str(train_loss/len(dataset_train)))
        
        val_loss = 0
        for i, data in enumerate(val_train, 0):
            val_truth, val_noise =  data
            val_output = model(val_noise.unsqueeze(1).float().to(args.device))
            output_loss = criterion(val_output.squeeze(1).to(args.device), val_truth.to(args.device), 10).to(args.device)
            val_loss+=output_loss.item()
        scheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss/len(val_train)
        print("Validation: "+ str(val_loss/len(val_train)))
        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net_721_1.pth'))
    training = plt.plot(training_losses, label='training')
    validation = plt.plot(validation_losses, label='validation')
    plt.legend()
    plt.savefig("lossplt_721_1.png")

    #make some images and store to csv
    
    branch = get_all_histograms("test.root")
    for image in range(10):
        model.to('cpu')
        data = get_bin_weights(branch, image).copy()
        np.savetxt('logs/7_21_1_truth#' + str(image) + '.txt', data)
        noisy = add_noise(data, args.sigma).copy()
        np.savetxt('logs/7_21_1_noised#' + str(image) + '.txt', noisy)
        data = torch.from_numpy(data)
        noisy = torch.from_numpy(noisy)
        noisy = noisy.unsqueeze(0)
        noisy = noisy.unsqueeze(1)
        output = model(noisy.float()).squeeze(0).squeeze(0).detach().numpy()
        np.savetxt('logs/7_21_1_denoised#' + str(image) + '.txt', output)
    
    
if __name__ == "__main__":
    main()
