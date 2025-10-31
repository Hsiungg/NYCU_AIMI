import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import dataloader
from models.EEGNet import EEGNet, DeepConvNet
from utils import set_seed


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index, ...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0


def build_model(args) -> nn.Module:
    if args.model == "EEGNet":
        return EEGNet(args)
    elif args.model == "DeepConvNet":
        return DeepConvNet(args)
    else:
        raise ValueError(f"Invalid model: {args.model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, default=512)
    parser.add_argument("-activation_function", type=str, default="relu", choices=["relu", "leakyrelu", "elu", "selu"]) 
    parser.add_argument("-elu_alpha", type=float, default=0.1)
    parser.add_argument("-dropout_rate", type=float, default=0.1)
    parser.add_argument("-model", type=str, default="EEGNet", choices=["EEGNet", "DeepConvNet"])
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-weights", type=str, default="", help="Path to weights .pt file. If empty, infer from naming.")
    parser.add_argument("-seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Deterministic evaluation to avoid run-to-run fluctuations
    set_seed(args.seed)

    # Prepare data
    _, _, test_data, test_label = dataloader.read_bci_data()
    test_dataset = BCIDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Build and load model
    model = build_model(args)
    model.to(device)

    weight_path = args.weights
    if not weight_path:
        weight_path = (
            f"./weights/{args.model}_{args.activation_function}_alpha{args.elu_alpha}_"
            f"dropout{args.dropout_rate}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}.pt"
        )

    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    # Evaluate
    acc = evaluate(model, test_loader, device)
    print(f"Test Acc. (%): {acc:3.2f}%")


if __name__ == "__main__":
    main()


