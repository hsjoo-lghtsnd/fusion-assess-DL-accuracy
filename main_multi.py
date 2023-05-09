import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from sklearn.metrics import *
from matplotlib import pyplot as plt

from datetime import datetime
from pytictoc import TicToc

import cifar10_loader
from fusion import fusion
''' quality assessment is done before '''

OPTIONS = 7

def _now():
    now = datetime.now()
    timetag = now.strftime('%Y%m%d-%H%M%S')
    return timetag

class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1,   16,  3).to(device)
        self.conv2 = nn.Conv2d(16,  64,  3).to(device)
        self.conv3 = nn.Conv2d(64,  128, 3).to(device)
        self.conv4 = nn.Conv2d(128, 256, 3).to(device)

        self.pool = nn.MaxPool2d(2, 2).to(device)

        self.fc1 = nn.Linear(256 * 2 * 2, 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.fc3 = nn.Linear(64, 10).to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    losses = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx > 0 and batch_idx % 2000 == 0:
            print('Train Epoch: {} [{}/{}\t({:.0f}%)]\tLoss {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return losses

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    if epoch > -1:
        print('epoch '+epoch)
    else:
        print('Test!')
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (float(correct) / len(test_loader.dataset))


def label_converter(labels):
    N, C = labels.shape
    classes = range(C)
    return np.einsum('i,ji->j', classes, labels)

def make_torchLoader(fim, lab, device):
    my_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(fim), torch.from_numpy(lab)),
            batch_size=8,
            shuffle=True,
            num_workers=2)
    for dat, lab in my_loader:
        dat = dat.to(device, non_blocking=True)
        lab = lab.to(device, dtype=torch.uint8, non_blocking=True)
    return my_loader

def dev_fim_loader_path(im, lab, opt, dirpath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fim = fusion.baseline(im, opt)
    fim = fim.reshape(-1, 1, 32, 32)

    my_loader = make_torchLoader(fim, lab, device)
    path = os.path.join(dirpath, 'model' + str(opt) + '.pt')
    return device, fim, my_loader, path


def fusion_train_save(im, lab, opt, dirpath):
    device, fim, my_loader, path = dev_fim_loader_path(im, lab, opt, dirpath)

    model = CNN(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    print(_now() + ':training on ' + path)

    losses = []
    accuracies = []
    for epoch in range(0, 30):
        losses.extend(train(model, device, my_loader, optimizer, epoch))
        accuracies.append(test(model, device, my_loader, epoch))

    torch.save(model.state_dict(), path)
    print(_now() + ':save completed on ' + path)

    return

def print_test(im, lab, opt, dirpath):
    device, fim, my_loader, path = dev_fim_loader_path(im, lab, opt, dirpath)
   
    model = CNN(device)
    model.load_state_dict(torch.load(path))

    print ('loading ' + path)
    return test(model, device, my_loader, -1)



def my_main(MULTI):
    if MULTI:
        func = 2
    else:
        func = int(input('[0]: train, [1]: test, [2]: both, [3]: fig\n>>> '))

    t = TicToc()

    t.tic()
    (train_data, train_label, test_data, test_label) = cifar10_loader.cifar10('./data/cifar10')
    print('cifar10 load completed')
    t.toc()

    train_data = train_data.reshape(-1, 3, 32, 32)
    test_data = test_data.reshape(-1, 3, 32, 32)

    train_lab = label_converter(train_label)
    test_lab = label_converter(test_label)

    if func == 0:
        path = os.path.join('.', 'trained', _now())
        os.makedirs(path, exist_ok=True)

        for opt in range(OPTIONS):
            fusion_train_save(train_data, train_lab, opt, path)

    elif func == 1:
        path = os.path.join('.', 'trained')

        for opt in range(OPTIONS):
            print_test(test_data, test_lab, opt, path)

    elif func == 2:
        path = os.path.join('.', 'trained', _now())
        os.makedirs(path, exist_ok=True)

        for opt in range(OPTIONS):
            fusion_train_save(train_data, train_lab, opt, path)
            print_test(test_data, test_lab, opt, path)

    return

if __name__ == "__main__":
    for i in range(30):
        my_main(MULTI=True)

