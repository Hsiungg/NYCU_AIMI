import copy
import torch
import argparse
import dataloader
import numpy as np
import random
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
from utils import set_seed, grid_search
from plotting import save_parallel_coordinates


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


def train(model, loader, criterion, optimizer, args, writer=None, verbose=True):
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
            _iter = tqdm(loader) if verbose else loader
            for i, data in enumerate(_iter, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            if verbose:
                print(f'Epoch: {epoch}')
                print(f'Loss: {avg_loss}')
                print(f'Training Acc. (%): {avg_acc:3.2f}%')
            if writer is not None and verbose:
                writer.add_scalar('train/epoch_loss', avg_loss, epoch)
                writer.add_scalar('train/epoch_acc', avg_acc, epoch)

        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = deepcopy(model.state_dict())
        if verbose:
            print(f'Test Acc. (%): {test_acc:3.2f}%')
        if writer is not None and verbose:
            writer.add_scalar('test/epoch_acc', test_acc, epoch)

    if verbose:
        print(f'Best Test Acc. (%): {best_acc:3.2f}%')
    # Ensure weights directory exists
    os.makedirs('./weights', exist_ok=True)
    weight_name = (
        f"{args.model}_{args.activation_function}_alpha{args.elu_alpha}_"
        f"dropout{args.dropout_rate}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}.pt"
    )
    weight_path = os.path.join('./weights', weight_name)
    if verbose:
        print(f'Best model saved to: {weight_path}')
    torch.save(best_wts, weight_path)
    return avg_acc_list, avg_loss_list, test_acc_list


def test(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc


def _parse_list(csv: str, cast):
    return [cast(x.strip()) for x in csv.split(',') if x.strip() != '']


def _run_once(config, base_args):
    # Copy base args and override with config
    args = deepcopy(base_args)
    for k, v in config.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # Reproducibility
    set_seed(getattr(args, 'seed', 0))

    # Build loaders for this run
    global train_loader, test_loader
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build model per config
    if args.model == "EEGNet":
        model = EEGNet(args)
    elif args.model == "DeepConvNet":
        model = DeepConvNet(args)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=getattr(args, 'weight_decay', 0.001))

    model.to(device)
    criterion.to(device)

    # Train without TensorBoard for grid search
    _, _, test_acc_list = train(
        model, train_loader, criterion, optimizer, args, writer=None, verbose=False)
    best_acc = max(test_acc_list) if len(test_acc_list) > 0 else 0.0
    return {"metric": best_acc, "best_acc": best_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=300)
    parser.add_argument("-batch_size", type=int, default=512)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-log_dir", type=str, default="runs/lab2")
    parser.add_argument("-activation_function", type=str,
                        default="relu", choices=["relu", "leakyrelu", "elu", "selu"])
    parser.add_argument("-elu_alpha", type=float, default=0.1)
    parser.add_argument("-dropout_rate", type=float, default=0.1)
    parser.add_argument("-model", type=str, default="EEGNet",
                        choices=["EEGNet", "DeepConvNet"])
    parser.add_argument("-out_dir", type=str, default="outputs")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-num_workers", type=int, default=0)
    # Grid search options
    parser.add_argument("-grid_search", action="store_true")
    parser.add_argument("-grid_lr", type=str, default="0.01, 0.005, 0.001")
    parser.add_argument("-grid_dropout", type=str,
                        default="0.1, 0.2, 0.3, 0.4, 0.5")
    parser.add_argument("-grid_elu_alpha", type=str,
                        default="0.1, 0.2, 0.4, 0.6, 0.8, 1.0")
    parser.add_argument("-grid_num_epochs", type=int, default=300)
    parser.add_argument("-grid_batch_size", type=str,
                        default="64, 128, 256, 512")
    parser.add_argument("-grid_activation", type=str,
                        default="relu, leakyrelu, elu, selu")
    # removed unused -grid_weight_decay
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Reproducibility
    set_seed(args.seed)

    # If grid search requested, run it and exit
    if args.grid_search:
        # Base param grid
        param_grid = {
            "lr": _parse_list(args.grid_lr, float),
            "dropout_rate": _parse_list(args.grid_dropout, float),
            "elu_alpha": _parse_list(args.grid_elu_alpha, float),
            # Search batch size as well
            "batch_size": _parse_list(args.grid_batch_size, int),
            # weight_decay grid removed
            "model": [args.model],
            "num_epochs": [args.grid_num_epochs],
            "seed": [args.seed],
        }
        # Grid-search activation_function for both models
        param_grid["activation_function"] = _parse_list(
            args.grid_activation, str)
        best_config, best_result, all_results = grid_search(param_grid, lambda cfg: _run_once(
            cfg, args), metric_key="metric", maximize=True, verbose=True)
        print("BEST CONFIG:", best_config)
        print("BEST RESULT:", best_result)

        # ===== Save results and draw Parallel Coordinates plot =====
        try:
            import time
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_dir = os.path.join(args.out_dir, "grid_search", ts)
            os.makedirs(out_dir, exist_ok=True)

            # Flatten results into a DataFrame
            rows = []
            for cfg, res in all_results:
                row = dict(cfg)
                if isinstance(res, dict):
                    for k, v in res.items():
                        row[k] = v
                else:
                    row["metric"] = float(res)
                rows.append(row)
            df = pd.DataFrame(rows)

            # Persist raw results
            csv_path = os.path.join(out_dir, "grid_search_results.csv")
            df.to_csv(csv_path, index=False)

            # Create parallel coordinates plot with requested bins/palette
            save_parallel_coordinates(df, out_dir)

            print(f"Grid search artifacts saved to: {out_dir}")
        except Exception as e:
            print(f"Failed to save parallel coordinates plot: {e}")
        raise SystemExit(0)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # TODO write EEGNet yourself
    if args.model == "EEGNet":
        model = EEGNet(args)
    elif args.model == "DeepConvNet":
        model = DeepConvNet(args)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=getattr(args, 'weight_decay', 0.001))

    model.to(device)
    criterion.to(device)

    # TensorBoard writer: unique per-run directory to avoid merged curves
    from datetime import datetime
    run_name = (
        f"{args.model}_act-{args.activation_function}_alpha-{args.elu_alpha}_"
        f"dropout-{args.dropout_rate}_bs-{args.batch_size}_lr-{args.lr}_seed-{args.seed}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    tb_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(tb_dir, exist_ok=True)
    print(f"TensorBoard log_dir: {tb_dir}")
    writer = SummaryWriter(log_dir=tb_dir)

    train_acc_list, train_loss_list, test_acc_list = train(
        model, train_loader, criterion, optimizer, args, writer)

    writer.close()

    save_dir = os.path.join(
        args.out_dir,
        args.model,
        args.activation_function,
        f"alpha{args.elu_alpha}",
        f"dropout{args.dropout_rate}",
        f"bs{args.batch_size}",
        f"lr{args.lr}",
        f"seed{args.seed}",
    )
    plot_train_acc(train_acc_list, args.num_epochs, save_dir)
    plot_train_loss(train_loss_list, args.num_epochs, save_dir)
    plot_test_acc(test_acc_list, args.num_epochs, save_dir)
