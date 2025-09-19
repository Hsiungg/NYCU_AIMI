import os
from random import choices
import warnings
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn


def train(model_name, device, train_loader, val_loader, model, criterion, optimizer, writer):
    best_acc = 0.0
    stalled_epoch = 0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    train_f1_score_list = []
    val_f1_score_list = []
    best_c_matrix = []

    for epoch in range(1, args.num_epochs+1):

        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0     
            for _, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn          

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            train_f1_score = 2*tp / (2*tp+fp+fn)
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')
            print(f'↳ Training F1-score: {train_f1_score:.4f}')

        # write validation if you needed
        val_acc, val_f1_score, c_matrix = test(val_loader, model, device, mode = "Val")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('F1Score/Train', train_f1_score, epoch)
        writer.add_scalar('F1Score/Validation', val_f1_score, epoch)

        train_acc_list.append(train_acc)
        train_f1_score_list.append(train_f1_score)
        val_acc_list.append(val_acc)
        val_f1_score_list.append(val_f1_score)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            best_c_matrix = c_matrix
        else:
            stalled_epoch += 1
            if stalled_epoch > 10:
                print(f"No improvement for 10 epochs. Early stopping at epoch {epoch}!")
                break

    torch.save(best_model_wts, f'model_{model_name}_weights.pt')

    return train_acc_list, train_f1_score_list, val_acc_list, val_f1_score_list, best_c_matrix

def test(test_loader, model, device, mode = "Test"):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in tqdm(test_loader):
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ {mode} Recall: {recall:.4f}, {mode} Precision: {precision:.4f}, {mode} F1-score: {f1_score:.4f}')
        print (f'↳ {mode} Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)
    parser.add_argument('--model_name', type=str, required=True, choices=['resnet18', 'resnet34', 'resnet50'])

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set seed for reproducibility
    set_seed(42)

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader (Train and Test dataset, write your own validation dataloader if needed.)
    # TODO / Change the data augmentation method yourself
    train_transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                          transforms.RandomRotation(args.degree),
                                          transforms.ToTensor()])
    
    test_transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                          transforms.ToTensor()])
    train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'),
                                transform = train_transform)
    
    val_dataset = ImageFolder(root=os.path.join(args.dataset, 'val'),
                              transform = test_transform)

    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                               transform = test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Print dataset statistics
    print(f"\n## Dataset Statistics ##")
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")
    
    # Print class information
    print(f"\nClass names: {train_dataset.classes}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    # Count images per class in training set
    from collections import Counter
    train_targets = [train_dataset.targets[i] for i in range(len(train_dataset.targets))]
    train_class_counts = Counter(train_targets)
    print(f"\nTraining set class distribution:")
    for class_idx, count in train_class_counts.items():
        class_name = train_dataset.classes[class_idx]
        print(f"  {class_name}: {count} images")
    
    # Count images per class in validation set
    val_targets = [val_dataset.targets[i] for i in range(len(val_dataset.targets))]
    val_class_counts = Counter(val_targets)
    print(f"\nValidation set class distribution:")
    for class_idx, count in val_class_counts.items():
        class_name = val_dataset.classes[class_idx]
        print(f"  {class_name}: {count} images")
    
    # Count images per class in test set
    test_targets = [test_dataset.targets[i] for i in range(len(test_dataset.targets))]
    test_class_counts = Counter(test_targets)
    print(f"\nTest set class distribution:")
    for class_idx, count in test_class_counts.items():
        class_name = test_dataset.classes[class_idx]
        print(f"  {class_name}: {count} images")

    # TODO / define model
    if args.model_name == 'resnet18':
        model = models.resnet18(weights="DEFAULT")
    elif args.model_name == 'resnet34':
        model = models.resnet34(weights="DEFAULT")
    elif args.model_name == 'resnet50':
        model = models.resnet50(weights="DEFAULT")
    
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Initialize TensorBoard writer
    log_dir = f'runs/{args.model_name}_{args.dataset}_lr{args.lr}_bs{args.batch_size}'
    writer = SummaryWriter(log_dir)

    # training
    train_acc_list, train_f1_score_list, val_acc_list, val_f1_score_list, best_c_matrix = train(args.model_name, device, train_loader, val_loader, model, criterion, optimizer, writer)
    # testing
    print("Testing best model")
    model.load_state_dict(torch.load(f'model_{args.model_name}_weights.pt'))
    test_acc, test_f1_score, test_c_matrix = test(test_loader, model, device, mode = "Test")
    
    # Log final test results to TensorBoard
    writer.add_scalar('Test/Accuracy', test_acc, 0)
    writer.add_scalar('Test/F1Score', test_f1_score, 0)
    
    # Log confusion matrix as image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(test_c_matrix, cmap='Blues')
    
    # Set class names as tick labels
    class_names = test_dataset.classes
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Test Confusion Matrix')
    
    # Add text annotations
    for i in range(len(test_c_matrix)):
        for j in range(len(test_c_matrix[0])):
            ax.text(j, i, str(test_c_matrix[i][j]), ha='center', va='center',
                   color="white" if test_c_matrix[i][j] > np.max(test_c_matrix)/2 else "black")
    
    writer.add_figure('Test/Confusion_Matrix', fig, 0)
    
    # Close the writer
    writer.close()
    
    print(f"\nTensorBoard logs saved to: {log_dir}")
    print(f"To view results, run: tensorboard --logdir={log_dir}")