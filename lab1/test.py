import os
import warnings
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import logging
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

def setup_logging(model_name, dataset):
    """Setup logging configuration for testing"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_log_{model_name}_{dataset}_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_path

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

def test_model(test_loader, model, device, class_names):
    """Test the model and return detailed metrics"""
    model.eval()
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    all_predictions = []
    all_labels = []
    
    with torch.set_grad_enabled(False):
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.max(outputs, 1).indices

            # Store for detailed analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(predictions, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        # Calculate metrics
        accuracy = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn) if (tp+fn) > 0 else 0
        precision = tp / (tp+fp) if (tp+fp) > 0 else 0
        f1_score = (2*tp) / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
        
        c_matrix = [[int(tn), int(fp)],
                    [int(fn), int(tp)]]

        # Log results
        logging.info("=== Test Results ===")
        logging.info(f"Accuracy: {accuracy:.2f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1_score:.3f}")
        logging.info(f"Confusion Matrix: TP={int(tp)}, TN={int(tn)}, FP={int(fp)}, FN={int(fn)}")

    return accuracy, f1_score, c_matrix, all_predictions, all_labels

def plot_confusion_matrix(c_matrix, class_names, save_path=None):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(c_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Test Confusion Matrix')
    plt.colorbar()
    
    # Add class names as tick labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    for i in range(len(c_matrix)):
        for j in range(len(c_matrix[0])):
            plt.text(j, i, str(c_matrix[i][j]), 
                    horizontalalignment="center", verticalalignment="center",
                    color="white" if c_matrix[i][j] > np.max(c_matrix)/2 else "black",
                    fontsize=14, fontweight='bold')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to: {save_path}")
    
    plt.show()

def analyze_predictions(all_predictions, all_labels, class_names):
    """Analyze prediction patterns"""
    logging.info("=== Per-Class Results ===")
    
    # Calculate per-class accuracy
    correct_per_class = {0: 0, 1: 0}
    total_per_class = {0: 0, 1: 0}
    
    for pred, true in zip(all_predictions, all_labels):
        total_per_class[true] += 1
        if pred == true:
            correct_per_class[true] += 1
    
    for class_idx in range(len(class_names)):
        if total_per_class[class_idx] > 0:
            acc = correct_per_class[class_idx] / total_per_class[class_idx] * 100
            logging.info(f"{class_names[class_idx]}: {acc:.1f}% ({correct_per_class[class_idx]}/{total_per_class[class_idx]})")

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # Model and dataset arguments
    parser.add_argument('--model_name', type=str, required=True, help='Model name (resnet18, resnet34, resnet50)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray', help='Dataset path')
    parser.add_argument('--num_classes', type=int, required=False, default=2, help='Number of classes')
    
    # Test arguments
    parser.add_argument('--batch_size', type=int, required=False, default=1, help='Batch size for testing')
    parser.add_argument('--resize', type=int, default=384, help='Image resize dimension')
    parser.add_argument('--save_plots', action='store_true', help='Also display confusion matrix plot (always saves to file)')

    args = parser.parse_args()

    # Setup logging
    log_path = setup_logging(args.model_name, args.dataset)
    logging.info(f"Testing started. Log file: {log_path}")
    
    # Log test configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Testing {args.model_name} on {device} | Dataset: {args.dataset} | Batch: {args.batch_size}")

    # Load test dataset
    test_transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.CenterCrop(352),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4823, 0.4823, 0.4823], std=[0.2363, 0.2363, 0.2363])
        ])

    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                              transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Log dataset statistics
    test_targets = [test_dataset.targets[i] for i in range(len(test_dataset.targets))]
    test_class_counts = Counter(test_targets)
    class_dist = " | ".join([f"{test_dataset.classes[idx]}: {count}" for idx, count in test_class_counts.items()])
    logging.info(f"Test set: {len(test_dataset)} images ({class_dist})")

    # Load model
    logging.info("=== Loading Model ===")
    if args.model_name == 'resnet18':
        model = models.resnet18(weights=None)
    elif args.model_name == 'resnet34':
        model = models.resnet34(weights=None)
    elif args.model_name == 'resnet50':
        model = models.resnet50(weights=None)
    elif args.model_name == 'swin_v2':
        model = models.swin_v2_b(weights=None)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")
    
    # Modify final layer
    if args.model_name == 'swin_v2':
        model.head = nn.Linear(model.head.in_features, args.num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    
    # Load trained weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    logging.info(f"Model loaded successfully from: {args.model_path}")

    # Test the model
    logging.info("=== Starting Testing ===")
    accuracy, f1_score, c_matrix, all_predictions, all_labels = test_model(
        test_loader, model, device, test_dataset.classes
    )

    # Analyze predictions
    analyze_predictions(all_predictions, all_labels, test_dataset.classes)

    # Create results directory and save confusion matrix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"test_results_{args.model_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save confusion matrix
    plot_path = os.path.join(results_dir, f"confusion_matrix_{args.model_name}.png")
    plot_confusion_matrix(c_matrix, test_dataset.classes, plot_path)
    
    # Also show the plot if save_plots is enabled
    if args.save_plots:
        plot_confusion_matrix(c_matrix, test_dataset.classes)

    logging.info("=== Testing Completed ===")
    logging.info(f"Final Test Accuracy: {accuracy:.2f}%")
    logging.info(f"Final F1-Score: {f1_score:.4f}")
    logging.info(f"Results saved to: {results_dir}/")
