#!/usr/bin/env python3
"""
Train ViT, Swin, or TIMM models for pneumonia classification
Supports data resampling for imbalanced datasets
Output format compatible with Kaggle submission requirements
"""
import os
import sys
import subprocess

def setup_cuda_environment():
    """Setup CUDA environment variables to ensure nvcc can be found"""
    if 'CUDA_HOME' in os.environ:
        cuda_home = os.environ['CUDA_HOME']
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
        if os.path.exists(nvcc_path):
            return
    
    possible_paths = [
        '/usr/lib/nvidia-cuda-toolkit',
        '/usr/local/cuda',
    ]
    
    for cuda_home in possible_paths:
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
        if os.path.exists(nvcc_path):
            os.environ['CUDA_HOME'] = cuda_home
            os.environ['PATH'] = f"{os.path.join(cuda_home, 'bin')}:{os.environ.get('PATH', '')}"
            print(f"Set CUDA_HOME: {cuda_home}")
            return
    
    try:
        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
        if result.returncode == 0:
            nvcc_path = result.stdout.strip()
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
            os.environ['CUDA_HOME'] = cuda_home
            print(f"Found nvcc in PATH, set CUDA_HOME: {cuda_home}")
            return
    except:
        pass
    
    print("Warning: nvcc not found, deepspeed may not work properly")

setup_cuda_environment()

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import random
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

try:
    import timm
    from timm.data import create_transform
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")

LABELS = ['normal', 'bacteria', 'virus', 'COVID-19']
NUM_CLASSES = len(LABELS)


class PneumoniaDataset(Dataset):
    """Pneumonia classification dataset"""
    
    def __init__(self, csv_path, img_dir, processor=None, transform=None, is_training=False, use_timm=False):
        """
        Args:
            csv_path: CSV file path
            img_dir: Image directory
            processor: HuggingFace image processor (for HF models)
            transform: Data augmentation transforms (for timm models)
            is_training: Whether in training mode
            use_timm: Whether using timm model
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.transform = transform
        self.is_training = is_training
        self.use_timm = use_timm
        
        self.labels = []
        for _, row in self.df.iterrows():
            label_vec = [row[label] for label in LABELS]
            label_idx = np.argmax(label_vec)
            self.labels.append(label_idx)
        
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['new_filename']
        img_path = self.img_dir / filename
        
        if not img_path.exists():
            base_path = img_path.with_suffix('')
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidate = base_path.with_suffix(ext)
                if candidate.exists():
                    img_path = candidate
                    break
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.use_timm:
            if self.transform:
                image = self.transform(image)
            return {
                'image': image,
                'labels': torch.tensor(self.labels[idx], dtype=torch.long),
                'idx': idx,
            }
        else:
            if self.transform:
                image = self.transform(image)
            
            inputs = self.processor(image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            
            return {
                'pixel_values': pixel_values,
                'labels': torch.tensor(self.labels[idx], dtype=torch.long),
                'idx': idx,
            }


def calculate_class_weights(dataset):
    """Calculate class weights for handling imbalanced data"""
    labels = dataset.labels
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"\nClass distribution:")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {class_counts[i]} samples, weight: {class_weights[i]:.4f}")
    
    return class_weights


def create_weighted_sampler(dataset):
    """Create weighted sampler for resampling"""
    labels = dataset.labels
    class_weights = calculate_class_weights(dataset)
    
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    try:
        labels_binary = label_binarize(labels, classes=range(NUM_CLASSES))
        predictions_proba = torch.softmax(torch.tensor(eval_pred.predictions), dim=-1).numpy()
        
        if labels_binary.shape[1] == 1:
            auc = roc_auc_score(labels, predictions_proba[:, 1])
        else:
            auc = roc_auc_score(labels_binary, predictions_proba, average='macro', multi_class='ovr')
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_macro': auc,
    }


class CustomTrainer(Trainer):
    """Custom Trainer with class weights support"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            if torch.cuda.is_available():
                self.class_weights = self.class_weights.cuda()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {k: v for k, v in inputs.items() if k not in ['idx']}
        labels = model_inputs.pop("labels")
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class TIMMTrainer:
    """Custom trainer for TIMM models"""
    
    def __init__(self, model, train_dataset, val_dataset, class_weights=None, 
                 train_sampler=None, output_dir='./output', num_epochs=10, 
                 batch_size=16, learning_rate=2e-5, early_stopping_patience=10,
                 device='cuda'):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.class_weights = class_weights
        self.train_sampler = train_sampler
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        
        self.model.to(device)
        
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        self.best_f1 = 0.0
        self.patience_counter = 0
        
    def get_dataloader(self, dataset, shuffle=False, sampler=None):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        try:
            labels_binary = label_binarize(all_labels, classes=range(NUM_CLASSES))
            all_probs = []
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    outputs = self.model(images)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
            all_probs = np.array(all_probs)
            
            if labels_binary.shape[1] == 1:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(labels_binary, all_probs, average='macro', multi_class='ovr')
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auc_macro': auc,
        }
    
    def train(self):
        train_loader = self.get_dataloader(self.train_dataset, shuffle=False, sampler=self.train_sampler)
        val_loader = self.get_dataloader(self.val_dataset, shuffle=False)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            eval_results = self.evaluate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {eval_results['accuracy']:.4f}")
            print(f"Val F1 Macro: {eval_results['f1_macro']:.4f}")
            print(f"Val F1 Weighted: {eval_results['f1_weighted']:.4f}")
            print(f"Val AUC: {eval_results['auc_macro']:.4f}")
            
            self.scheduler.step()
            
            if eval_results['f1_macro'] > self.best_f1:
                self.best_f1 = eval_results['f1_macro']
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
                print(f"Saved best model (F1: {self.best_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.model.load_state_dict(torch.load(self.output_dir / 'best_model.pth'))
        return self.model
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.output_dir / 'final_model.pth')
        label_map = {i: label for i, label in enumerate(LABELS)}
        with open(self.output_dir / 'label_map.json', 'w') as f:
            json.dump(label_map, f, indent=2)


def train_model(
    model_name='google/vit-base-patch16-224',
    train_csv='train_data.csv',
    val_csv='val_data.csv',
    train_img_dir='train_images',
    val_img_dir='val_images',
    output_dir='./output/hf_model',
    num_epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    use_resample=True,
    use_class_weights=True,
    model_type='vit',
    early_stopping_patience=10,
    no_early_stopping=False,
    seed=42,
):
    """Train model"""
    
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_seed(seed)
    print(f"Set random seed: {seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    use_timm = model_name.startswith('timm/') or 'eva02' in model_name.lower() or 'dinov3' in model_name.lower()
    
    if use_timm:
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for TIMM models. Install with: pip install timm")
        
        timm_model_name = model_name.replace('timm/', '') if model_name.startswith('timm/') else model_name
        print(f"\nLoading TIMM model: {timm_model_name}")
        
        model = timm.create_model(
            timm_model_name,
            pretrained=True,
            num_classes=NUM_CLASSES,
        )
        
        # Custom transforms for specific models
        if 'eva02' in timm_model_name.lower():
            # EVA02: Resize(512), CenterCrop(448), RandomHorizontalFlip, ToTensor, Normalize
            train_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(448),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # X-ray grayscale (RGB channels)
            ])
            val_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            print("\nUsing custom transforms for EVA02: Resize(512), CenterCrop(448), RandomHorizontalFlip, ToTensor, Normalize")
        elif 'dinov3' in timm_model_name.lower():
            # DINOv3: 256x256 transforms (X-ray specific normalization)
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # X-ray grayscale
            ])
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            print("\nUsing custom transforms for DINOv3: Resize(256), CenterCrop(256), X-ray normalization")
        else:
            # Default TIMM transforms
            data_config = timm.data.resolve_data_config(model.pretrained_cfg)
            train_transform = create_transform(
                **data_config,
                is_training=True,
            )
            val_transform = create_transform(
                **data_config,
                is_training=False,
            )
            print("\nUsing TIMM default data transforms")
        
        processor = None
        
    else:
        print(f"\nLoading HuggingFace model: {model_name}")
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        val_transform = None
        print("\nUsing data augmentation: RandomHorizontalFlip, RandomRotation, RandomAffine, ColorJitter")
    
    print("\nLoading datasets...")
    train_dataset = PneumoniaDataset(
        csv_path=train_csv,
        img_dir=train_img_dir,
        processor=processor,
        transform=train_transform,
        is_training=True,
        use_timm=use_timm,
    )
    
    val_dataset = PneumoniaDataset(
        csv_path=val_csv,
        img_dir=val_img_dir,
        processor=processor,
        transform=val_transform,
        is_training=False,
        use_timm=use_timm,
    )
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples")
    
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(train_dataset)
        class_weights = class_weights.tolist()
    
    if not use_timm:
        print(f"\nLoading pretrained model: {model_name}")
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        )
    
    if use_timm:
        train_sampler = None
        if use_resample:
            train_sampler = create_weighted_sampler(train_dataset)
            print("\nUsing weighted resampling")
        
        trainer = TIMMTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            class_weights=class_weights,
            train_sampler=train_sampler,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience if not no_early_stopping else num_epochs,
            device=device,
        )
        
        print("\nStarting training...")
        trainer.train()
        
        print("\nEvaluating model...")
        val_loader = trainer.get_dataloader(val_dataset, shuffle=False)
        eval_results = trainer.evaluate(val_loader)
        print(f"\nValidation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\nSaving model to: {output_dir}")
        trainer.save_model()
        
        return trainer, model, None
    
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            report_to="none",
            seed=seed,
            data_seed=seed,
        )
        
        train_sampler = None
        if use_resample:
            train_sampler = create_weighted_sampler(train_dataset)
            print("\nUsing weighted resampling")
        
        class ResampleTrainer(CustomTrainer):
            def __init__(self, train_sampler=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.train_sampler = train_sampler
            
            def get_train_dataloader(self):
                if self.train_sampler is not None:
                    return DataLoader(
                        self.train_dataset,
                        batch_size=self.args.per_device_train_batch_size,
                        sampler=self.train_sampler,
                        num_workers=self.args.dataloader_num_workers,
                        pin_memory=True,
                    )
                else:
                    return super().get_train_dataloader()
        
        callbacks = []
        if not no_early_stopping:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
            print(f"\nUsing early stopping, patience={early_stopping_patience}")
        else:
            print("\nEarly stopping disabled, will train for full epochs")
        
        trainer = ResampleTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            train_sampler=train_sampler,
            callbacks=callbacks,
        )
        
        print("\nStarting training...")
        trainer.train()
        
        print("\nEvaluating model...")
        eval_results = trainer.evaluate()
        print(f"\nValidation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\nSaving model to: {output_dir}")
        trainer.save_model()
        processor.save_pretrained(output_dir)
        
        label_map = {i: label for i, label in enumerate(LABELS)}
        with open(f'{output_dir}/label_map.json', 'w') as f:
            json.dump(label_map, f, indent=2)
        
        print("\nTraining complete!")
        return trainer, model, processor


def main():
    parser = argparse.ArgumentParser(description='Train ViT/Swin/TIMM models for pneumonia classification')
    
    parser.add_argument('--train_csv', type=str, default='train_data.csv',
                        help='Training CSV file path')
    parser.add_argument('--val_csv', type=str, default='val_data.csv',
                        help='Validation CSV file path')
    parser.add_argument('--train_img_dir', type=str, default='train_images',
                        help='Training image directory')
    parser.add_argument('--val_img_dir', type=str, default='val_images',
                        help='Validation image directory')
    
    model_type = 'timm'
    parser.add_argument('--model_type', type=str, default=model_type,
                        choices=['vit', 'swin', 'timm'],
                        help='Model type: vit, swin, or timm')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Pretrained model name (optional, defaults based on model_type)')
    
    parser.add_argument('--output_dir', type=str, default=f"./output/hf_model_{model_type}",
                        help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    
    parser.add_argument('--use_resample', action='store_true', default=True,
                        help='Use resampling')
    parser.add_argument('--no_resample', dest='use_resample', action='store_false',
                        help='Disable resampling')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights')
    parser.add_argument('--no_class_weights', dest='use_class_weights', action='store_false',
                        help='Disable class weights')
    
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Disable early stopping, train for full epochs')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.model_name is None:
        if args.model_type == 'vit':
            args.model_name = 'google/vit-base-patch16-224'
        elif args.model_type == 'swin':
            args.model_name = 'microsoft/swin-base-patch4-window7-224'
        elif args.model_type == 'timm':
            args.model_name = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
    
    if args.output_dir is None:
        args.output_dir = f'./output/{args.model_type}_model'
    
    trainer, model, processor = train_model(
        model_name=args.model_name,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        train_img_dir=args.train_img_dir,
        val_img_dir=args.val_img_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_resample=args.use_resample,
        use_class_weights=args.use_class_weights,
        model_type=args.model_type,
        early_stopping_patience=args.early_stopping_patience,
        no_early_stopping=args.no_early_stopping,
        seed=args.seed,
    )
    
    print("\nTraining complete! Use predict.py script to make predictions on test set.")


if __name__ == '__main__':
    main()
