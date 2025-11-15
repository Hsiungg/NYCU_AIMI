#!/usr/bin/env python3
"""
Predict on test set using trained model
Output format compatible with Kaggle submission requirements
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import platform

# HuggingFace Transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Class labels (must match training)
LABELS = ['normal', 'bacteria', 'virus', 'COVID-19']
NUM_CLASSES = len(LABELS)

# Detect OS for DataLoader settings
IS_WINDOWS = platform.system() == 'Windows'
PREDICT_NUM_WORKERS = 0 if IS_WINDOWS else 4  # Windows: disable multiprocessing to avoid issues


class PneumoniaDataset(Dataset):
    """Pneumonia classification dataset for prediction"""
    
    def __init__(self, csv_path, img_dir, processor=None, transform=None, use_timm=False):
        """
        Args:
            csv_path: CSV file path
            img_dir: Image directory
            processor: HuggingFace image processor (optional)
            transform: Custom transforms for TIMM models (optional)
            use_timm: Whether using TIMM model
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.transform = transform
        self.use_timm = use_timm
        
        # Pre-import ToTensor to avoid issues with multiprocessing
        self.to_tensor = transforms.ToTensor()
        
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
            # TIMM models: use transform (includes ToTensor, Normalize)
            if self.transform:
                image = self.transform(image)
            else:
                # If no transform, at least convert to tensor
                image = self.to_tensor(image)
            return {
                'image': image,
                'idx': idx,
            }
        else:
            # HuggingFace models
            if self.transform:
                image = self.transform(image)
            
            # If processor is None, transform already returns a tensor
            if self.processor is None:
                pixel_values = image  # Already a tensor from transform
                # Safety check: if still PIL Image, convert to tensor
                if not isinstance(pixel_values, torch.Tensor):
                    pixel_values = self.to_tensor(pixel_values)
            else:
                inputs = self.processor(image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)
            
            return {
                'pixel_values': pixel_values,
                'idx': idx,
            }


def predict_test_set(
    checkpoint_path,
    test_csv='test_data.csv',
    test_img_dir='test_images',
    output_csv='test_data_predictions.csv',
    batch_size=16,
    device='cuda'
):
    """Predict on test set and update CSV"""
    
    checkpoint_path = Path(checkpoint_path)
    
    # If checkpoint_path points to a .pth file, use its parent directory
    if checkpoint_path.is_file() and checkpoint_path.suffix == '.pth':
        print(f"Detected .pth file, using parent directory: {checkpoint_path.parent}")
        checkpoint_path = checkpoint_path.parent
    
    print(f"\nLoading model from: {checkpoint_path}")
    
    # Check if it's a TIMM model (has .pth files but no config.json)
    has_pth = (checkpoint_path / 'best_model.pth').exists() or (checkpoint_path / 'final_model.pth').exists()
    has_config = (checkpoint_path / 'config.json').exists()
    
    # Load config to detect model type
    config_path = checkpoint_path / 'config.json'
    model_name = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            model_name = config.get('_name_or_path', '')
            print(f"Detected model from config: {model_name}")
    
    # Detect model type
    use_timm = False
    transform = None
    processor = None
    
    # If has .pth but no config.json, it's a TIMM model
    if has_pth and not has_config:
        use_timm = True
        # Try to detect model name from directory name
        dir_name = checkpoint_path.name
        if 'dinov3' in dir_name.lower():
            model_name = 'vit_base_patch16_dinov3.lvd1689m'  # Default DINOv3 model
            print(f"Detected TIMM DINOv3 model from directory name")
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Direct resize to preserve full image
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif 'eva02' in dir_name.lower():
            model_name = 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
            print(f"Detected TIMM EVA02 model from directory name")
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    elif model_name:
        # Check if it's a TIMM model from config
        if 'eva02' in model_name.lower() and 'facebook' not in model_name.lower():
            use_timm = True
            print("Detected EVA02 model - using TIMM transforms")
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif 'dinov3' in model_name.lower() or 'vit_base_patch16_dinov3' in model_name.lower():
            use_timm = True
            print("Detected DINOv3 TIMM model - using custom transforms")
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Direct resize to preserve full image
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    
    # Load model
    if use_timm:
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for TIMM models. Install with: pip install timm")
        # Load TIMM model
        try:
            print(f"Loading TIMM model: {model_name}")
            model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
            # Load weights from checkpoint
            state_dict_path = checkpoint_path / 'best_model.pth'
            if not state_dict_path.exists():
                state_dict_path = checkpoint_path / 'final_model.pth'
            if state_dict_path.exists():
                state_dict = torch.load(state_dict_path, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"TIMM model loaded from: {state_dict_path}")
            else:
                raise FileNotFoundError(f"Cannot find model weights in {checkpoint_path}")
        except Exception as e:
            print(f"Error loading TIMM model: {e}")
            raise
    else:
        # Load HuggingFace model
        try:
            model = AutoModelForImageClassification.from_pretrained(str(checkpoint_path))
            print("HuggingFace model loaded successfully")
        except Exception as e:
            print(f"Error: Cannot load model from {checkpoint_path}")
            print(f"Please ensure the path is correct and contains model files")
            raise e
        
        # Try to load processor from multiple locations
        processor_paths = [
            checkpoint_path,  # Try checkpoint directory first
            checkpoint_path.parent,  # Try parent directory (main output dir)
        ]
        
        if model_name:
            processor_paths.append(model_name)
        
        for proc_path in processor_paths:
            try:
                if isinstance(proc_path, str) and not os.path.exists(proc_path):
                    # Try to load from HuggingFace hub
                    processor = AutoImageProcessor.from_pretrained(proc_path, use_fast=True)
                    print(f"Processor loaded from: {proc_path}")
                    break
                elif isinstance(proc_path, Path) and proc_path.exists():
                    processor = AutoImageProcessor.from_pretrained(str(proc_path), use_fast=True)
                    print(f"Processor loaded from: {proc_path}")
                    break
            except Exception as e:
                continue
        
        if processor is None:
            print("Warning: Cannot find processor, will use default transform")
            # Create default transform for models without processor (e.g., DINOv2)
            # Use 256x256 as default size (works for most vision models)
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            print("Using default transform: Resize(256, 256) + Normalize")
    
    # Check label mapping
    label_map_paths = [
        checkpoint_path / 'label_map.json',
        checkpoint_path.parent / 'label_map.json',
    ]
    
    for label_map_path in label_map_paths:
        if label_map_path.exists():
            with open(label_map_path, 'r') as f:
                saved_label_map = json.load(f)
            print(f"Loaded label mapping: {saved_label_map}")
            # Verify label order
            saved_labels = [saved_label_map[str(i)] for i in range(NUM_CLASSES)]
            if saved_labels != LABELS:
                print(f"Warning: Saved label order {saved_labels} differs from current LABELS {LABELS}")
            break
    
    print("\nLoading test set...")
    test_dataset = PneumoniaDataset(
        csv_path=test_csv,
        img_dir=test_img_dir,
        processor=processor,
        transform=transform,
        use_timm=use_timm,
    )
    
    print(f"Test set samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=PREDICT_NUM_WORKERS,
        pin_memory=True
    )
    
    if IS_WINDOWS:
        print("Running on Windows: using num_workers=0 to avoid multiprocessing issues")
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_filenames = []
    
    print("\nMaking predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # Handle different input formats for TIMM vs HuggingFace models
            if use_timm:
                # TIMM models use 'image' key
                images = batch['image'].to(device)
                logits = model(images)
            else:
                # HuggingFace models use 'pixel_values' key
                pixel_values = batch['pixel_values'].to(device)
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Get prediction probabilities
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_predictions.extend(probs)
            
            # Get filenames using idx from batch
            if 'idx' in batch:
                batch_indices = batch['idx'].cpu().numpy().tolist()
            else:
                # If no idx, calculate based on batch position
                start_idx = batch_idx * test_loader.batch_size
                batch_indices = range(start_idx, start_idx + len(preds))
            
            for idx in batch_indices:
                filename = test_dataset.df.iloc[idx]['new_filename']
                all_filenames.append(filename)
    
    # Update CSV
    print(f"\nUpdating {output_csv}...")
    df = pd.read_csv(test_csv)
    
    # Create prediction dictionary
    pred_dict = {}
    for filename, pred_probs in zip(all_filenames, all_predictions):
        pred_idx = int(np.argmax(pred_probs))
        label_map = {
            label: int(i == pred_idx)
            for i, label in enumerate(LABELS)
        }
        pred_dict[filename] = label_map
    
    # Update DataFrame
    updated_count = 0
    for idx, row in df.iterrows():
        filename = row['new_filename']
        if filename in pred_dict:
            label_map = pred_dict[filename]
            for label in LABELS:
                if label in df.columns:
                    df.at[idx, label] = int(label_map.get(label, 0))
            updated_count += 1
    
    # Ensure columns are integer type
    for col in LABELS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"Updated {output_csv}")
    print(f"Updated {updated_count} predictions")
    
    # Show prediction statistics
    print(f"\nPrediction statistics:")
    for label in LABELS:
        if label in df.columns:
            count = df[label].sum()
            print(f"  {label}: {count} samples")
    
    return output_csv


def main():
    model_name = 'dinov2'
    parser = argparse.ArgumentParser(description='Predict on test set using trained model')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Model checkpoint path (directory containing model files)')
    
    parser.add_argument('--test_csv', type=str, default='test_data.csv',
                        help='Test CSV file path')
    parser.add_argument('--test_img_dir', type=str, default='test_images',
                        help='Test image directory')
    
    parser.add_argument('--output_csv', type=str, default=f'{model_name}.csv',
                        help='Output CSV file path')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {args.checkpoint_path}")
    
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV file not found: {args.test_csv}")
    
    if not os.path.exists(args.test_img_dir):
        raise FileNotFoundError(f"Test image directory not found: {args.test_img_dir}")
    
    predict_test_set(
        checkpoint_path=args.checkpoint_path,
        test_csv=args.test_csv,
        test_img_dir=args.test_img_dir,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print("\nPrediction complete!")


if __name__ == '__main__':
    main()

