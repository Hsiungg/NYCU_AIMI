import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import EEGNet, DeepConvNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]

def plot_train_acc(train_acc_list, epochs, save_dir):
    plt.figure()
    x = list(range(1, len(train_acc_list) + 1))
    plt.plot(x, train_acc_list, label='Train Acc (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    try:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'train_accuracy.png'))
    except Exception:
        pass
    plt.close()

def plot_train_loss(train_loss_list, epochs, save_dir):
    plt.figure()
    x = list(range(1, len(train_loss_list) + 1))
    plt.plot(x, train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    try:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    except Exception:
        pass
    plt.close()

def plot_test_acc(test_acc_list, epochs, save_dir):
    plt.figure()
    x = list(range(1, len(test_acc_list) + 1))
    plt.plot(x, test_acc_list, label='Test Acc (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Testing Accuracy')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    try:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'test_accuracy.png'))
    except Exception:
        pass
    plt.close()

def train(model, loader, criterion, optimizer, args, writer=None):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    for epoch in range(1, args.num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')
            if writer is not None:
                writer.add_scalar('train/epoch_loss', avg_loss, epoch)
                writer.add_scalar('train/epoch_acc', avg_acc, epoch)

        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = model.state_dict()
        print(f'Test Acc. (%): {test_acc:3.2f}%')
        if writer is not None:
            writer.add_scalar('test/epoch_acc', test_acc, epoch)

    torch.save(best_wts, f'./weights/{args.model}_{args.activation_function}_alpha{args.elu_alpha}_dropout{args.dropout_rate}.pt')
    return avg_acc_list, avg_loss_list, test_acc_list


def test(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=150)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.01)
    parser.add_argument("-log_dir", type=str, default="runs/lab2")
    parser.add_argument("-activation_function", type=str, default="elu", choices=["relu", "leakyrelu", "elu", "selu"]) 
    parser.add_argument("-elu_alpha", type=float, default=1.0)
    parser.add_argument("-dropout_rate", type=float, default=0.2)
    parser.add_argument("-model", type=str, default="EEGNet", choices=["EEGNet", "DeepConvNet"])
    parser.add_argument("-out_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # TODO write EEGNet yourself
    if args.model == "EEGNet":
        model = EEGNet(args)
    elif args.model == "DeepConvNet":
        model = DeepConvNet(args)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    model.to(device)
    criterion.to(device)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    train_acc_list, train_loss_list, test_acc_list = train(model, train_loader, criterion, optimizer, args, writer)

    writer.close()

    save_dir = os.path.join(args.out_dir, args.model, args.activation_function, f"alpha{args.elu_alpha}", f"dropout{args.dropout_rate}")
    plot_train_acc(train_acc_list, args.num_epochs, save_dir)
    plot_train_loss(train_loss_list, args.num_epochs, save_dir)
    plot_test_acc(test_acc_list, args.num_epochs, save_dir)