#!/usr/bin/env python3
"""
Train ViT, Swin, or TIMM models for pneumonia classification
Supports data resampling for imbalanced datasets
Output format compatible with Kaggle submission requirements
"""
import os
import sys
import subprocess
import platform

# Detect OS and set appropriate num_workers for DataLoader
IS_WINDOWS = platform.system() == 'Windows'
NUM_WORKERS = 4 if IS_WINDOWS else 7  # Windows: use 2-4 workers, Linux: use 8
PERSISTENT_WORKERS = NUM_WORKERS > 0  # Enable persistent workers if using multiprocessing
print(f"Detected OS: {platform.system()}, using num_workers={NUM_WORKERS}, persistent_workers={PERSISTENT_WORKERS}")

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
    
    # nvcc not found - CUDA training will still work with PyTorch

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
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

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
        
        # Pre-import ToTensor to avoid issues with multiprocessing
        self.to_tensor = transforms.ToTensor()
        
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
            else:
                # If no transform, at least convert to tensor
                image = self.to_tensor(image)
            return {
                'image': image,
                'labels': torch.tensor(self.labels[idx], dtype=torch.long),
                'idx': idx,
            }
        else:
            if self.transform:
                image = self.transform(image)
            
            # If processor is None, transform already returns a tensor (includes ToTensor + Normalize)
            if self.processor is None:
                # Transform already includes ToTensor and Normalize
                pixel_values = image  # image is already a tensor
                # Safety check: if still PIL Image, convert to tensor
                if not isinstance(pixel_values, torch.Tensor):
                    pixel_values = self.to_tensor(pixel_values)
            else:
                # Processor handles resize, normalize, etc.
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
                 device='cuda', use_wandb=False):
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
        self.use_wandb = use_wandb
        
    def get_dataloader(self, dataset, shuffle=False, sampler=None):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=PERSISTENT_WORKERS,
        )
    
    def train_epoch(self, train_loader, epoch=0):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log to wandb during training
            if self.use_wandb and num_batches % 50 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch + num_batches / len(train_loader),
                })
        
        avg_loss = total_loss / len(train_loader)
        if self.use_wandb:
            wandb.log({
                'train/epoch_loss': avg_loss,
                'train/epoch': epoch + 1,
            })
        return avg_loss
    
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
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auc_macro': auc,
        }
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/accuracy': accuracy,
                'val/f1_macro': f1_macro,
                'val/f1_weighted': f1_weighted,
                'val/auc_macro': auc,
            })
        
        return results
    
    def train(self):
        train_loader = self.get_dataloader(self.train_dataset, shuffle=False, sampler=self.train_sampler)
        val_loader = self.get_dataloader(self.val_dataset, shuffle=False)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            train_loss = self.train_epoch(train_loader, epoch=epoch)
            eval_results = self.evaluate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {eval_results['accuracy']:.4f}")
            print(f"Val F1 Macro: {eval_results['f1_macro']:.4f}")
            print(f"Val F1 Weighted: {eval_results['f1_weighted']:.4f}")
            print(f"Val AUC: {eval_results['auc_macro']:.4f}")
            
            # Log epoch-level metrics to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_loss,
                    'val/accuracy': eval_results['accuracy'],
                    'val/f1_macro': eval_results['f1_macro'],
                    'val/f1_weighted': eval_results['f1_weighted'],
                    'val/auc_macro': eval_results['auc_macro'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                })
            
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
        
        # Save best metrics
        metrics = {
            'best_f1_macro': float(self.best_f1),
        }
        with open(self.output_dir / 'best_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)


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
    use_wandb=False,
    wandb_project='pneumonia-classification',
    wandb_run_name=None,
    wandb_entity=None,
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
    
    # Initialize wandb if requested
    if use_wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required but not installed. Install with: pip install wandb")
        
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            config={
                'model_name': model_name,
                'model_type': model_type,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'use_resample': use_resample,
                'use_class_weights': use_class_weights,
                'early_stopping_patience': early_stopping_patience,
                'no_early_stopping': no_early_stopping,
                'seed': seed,
            }
        )
        print(f"\n✓ WANDB initialized: project={wandb_project}, run={wandb_run_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if using TIMM models
    # Note: DINOv3 uses TIMM, DINOv2 uses HuggingFace
    use_timm = (model_name.startswith('timm/') or 
                ('eva02' in model_name.lower() and 'facebook' not in model_name.lower()) or
                (model_type == 'timm' and 'facebook' not in model_name.lower()) or
                model_type == 'dinov3')
    
    # DINOv2 from HuggingFace should use HuggingFace loader, not TIMM
    use_hf_dinov2 = (model_type == 'dinov2' and 'facebook' in model_name.lower()) or \
                     ('facebook/dinov2' in model_name.lower())
    
    if use_timm and not use_hf_dinov2:
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
        elif 'dinov3' in timm_model_name.lower() or model_type == 'dinov3':
            # DINOv3: Direct resize to 256x256 (no crop to preserve full image)
            # Note: Most chest X-rays are landscape ~1.43 aspect ratio
            # Using direct resize instead of CenterCrop to avoid losing ~26% of lung area
            train_transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Direct resize to 256x256 (preserves full image)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # X-ray grayscale normalization
            ])
            val_transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Direct resize to 256x256
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            print("\nUsing DINOv3: direct resize to 256x256 (no crop, preserves full lung area)")
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
        
        # Custom transforms for DINOv2 from HuggingFace
        if use_hf_dinov2 or 'facebook/dinov2' in model_name.lower():
            # DINOv2: disable processor's automatic processing, do everything manually
            processor.do_resize = False  # Disable automatic resize
            processor.do_center_crop = False  # Disable automatic center crop
            processor.do_normalize = False  # Disable automatic normalize, do it manually
            
            # Direct resize to 256x256 (no crop to preserve full chest X-ray image)
            # Note: Most chest X-rays are landscape ~1.43 aspect ratio
            # Using direct resize instead of CenterCrop to avoid losing ~26% of lung area
            train_transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Direct resize to 256x256 (preserves full image)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # X-ray grayscale normalization
            ])
            val_transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Direct resize to 256x256
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # X-ray grayscale normalization
            ])
            # Set processor to None since we're doing everything manually
            processor = None
            print("\nUsing DINOv2: direct resize to 256x256 (no crop, preserves full lung area)")
        elif 'swinv2' in model_name.lower() and '384' in model_name.lower():
            # Swin V2 with 384x384 resolution: disable processor, do manual resize
            # The 192to384 models are trained for 384x384 input
            processor.do_resize = False
            processor.do_center_crop = False
            processor.do_normalize = False
            processor = None  # Use manual transforms
            
            train_transform = transforms.Compose([
                transforms.Resize((384, 384)),  # Direct resize to 384x384 (no crop)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
            ])
            val_transform = transforms.Compose([
                transforms.Resize((384, 384)),  # Direct resize to 384x384
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print("\nUsing Swin V2 with 384x384 direct resize (no crop) + data augmentation")
        else:
            # Standard HuggingFace models (ViT, Swin V1): direct resize to 224x224 (no crop)
            processor.do_resize = False
            processor.do_center_crop = False
            processor.do_normalize = False
            processor = None  # Use manual transforms
            
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Direct resize to 224x224 (no crop)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
            ])
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Direct resize to 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print("\nUsing ViT/Swin with 224x224 direct resize (no crop) + data augmentation")
    
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
    
    if not use_timm or use_hf_dinov2:
        print(f"\nLoading pretrained model: {model_name}")
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        )
    
    if use_timm and not use_hf_dinov2:
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
            use_wandb=use_wandb,
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
        
        # Finish wandb run
        if use_wandb:
            wandb.finish()
            print("✓ WANDB run finished")
        
        return trainer, model, None
    
    else:
        # Determine reporting backend
        report_to = "wandb" if use_wandb else "none"
        
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
            dataloader_num_workers=NUM_WORKERS,
            dataloader_persistent_workers=PERSISTENT_WORKERS,
            report_to=report_to,
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
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                        persistent_workers=PERSISTENT_WORKERS,
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
        
        # Only save processor if it exists (None for models with custom transforms)
        if processor is not None:
            processor.save_pretrained(output_dir)
            print("Processor saved")
        else:
            print("No processor to save (using custom transforms)")
        
        label_map = {i: label for i, label in enumerate(LABELS)}
        with open(f'{output_dir}/label_map.json', 'w') as f:
            json.dump(label_map, f, indent=2)
        
        # Finish wandb run
        if use_wandb:
            wandb.finish()
            print("✓ WANDB run finished")
        
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
    
    model_type = 'dinov2'
    parser.add_argument('--model_type', type=str, default=model_type,
                        choices=['vit', 'swin', 'timm', 'dinov2', 'dinov3'],
                        help='Model type: vit, swin, timm, dinov2, or dinov3')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Pretrained model name (optional, defaults based on model_type)')
    
    parser.add_argument('--output_dir', type=str, default=f"./output/hf_model_{model_type}",
                        help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
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
    
    # WANDB arguments
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='pneumonia-classification',
                        help='WANDB project name (default: pneumonia-classification)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='WANDB run name (default: auto-generated from config)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WANDB entity/team name (optional)')
    
    args = parser.parse_args()
    
    if args.model_name is None:
        if args.model_type == 'vit':
            args.model_name = 'google/vit-base-patch16-224'
        elif args.model_type == 'swin':
            args.model_name = 'microsoft/swin-base-patch4-window7-224'  # Default Swin V1 model
        elif args.model_type == 'timm':
            args.model_name = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
        elif args.model_type == 'dinov2':
            args.model_name = 'facebook/dinov2-base'  # Default DINOv2 model
        elif args.model_type == 'dinov3':
            args.model_name = 'vit_base_patch16_dinov3.lvd1689m'  # TIMM DINOv3 model
    
    if args.output_dir is None:
        args.output_dir = f'./output/{args.model_type}_model'
    
    # Auto-generate wandb_run_name if not provided
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model_type}_{args.model_name.split('/')[-1]}_resample{args.use_resample}_weights{args.use_class_weights}_pat{args.early_stopping_patience}_seed{args.seed}"
    
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
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
    )
    
    print("\nTraining complete! Use predict.py script to make predictions on test set.")


if __name__ == '__main__':
    main()
