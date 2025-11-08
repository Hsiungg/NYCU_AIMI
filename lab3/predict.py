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
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json

# HuggingFace Transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Class labels (must match training)
LABELS = ['normal', 'bacteria', 'virus', 'COVID-19']
NUM_CLASSES = len(LABELS)


class PneumoniaDataset(Dataset):
    """Pneumonia classification dataset for prediction"""
    
    def __init__(self, csv_path, img_dir, processor):
        """
        Args:
            csv_path: CSV file path
            img_dir: Image directory
            processor: HuggingFace image processor
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.processor = processor
        
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
    print(f"\nLoading model from: {checkpoint_path}")
    
    # Load model
    try:
        model = AutoModelForImageClassification.from_pretrained(str(checkpoint_path))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error: Cannot load model from {checkpoint_path}")
        print(f"Please ensure the path is correct and contains model files")
        raise e
    
    # Try to load processor from multiple locations
    processor = None
    processor_paths = [
        checkpoint_path,  # Try checkpoint directory first
        checkpoint_path.parent,  # Try parent directory (main output dir)
    ]
    
    # Also try to get original model name from config
    try:
        config_path = checkpoint_path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Some models store the original model name in _name_or_path
                if '_name_or_path' in config:
                    processor_paths.append(config['_name_or_path'])
    except:
        pass
    
    for proc_path in processor_paths:
        try:
            if isinstance(proc_path, str) and not os.path.exists(proc_path):
                continue
            processor = AutoImageProcessor.from_pretrained(str(proc_path), use_fast=True)
            print(f"Processor loaded from: {proc_path}")
            break
        except:
            continue
    
    if processor is None:
        raise FileNotFoundError(
            f"Cannot find processor. Tried:\n" + 
            "\n".join([f"  - {p}" for p in processor_paths if isinstance(p, Path) or os.path.exists(str(p))])
        )
    
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
        processor=processor
    )
    
    print(f"Test set samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_filenames = []
    
    print("\nMaking predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            pixel_values = batch['pixel_values'].to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # 获取预测概率
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
    parser = argparse.ArgumentParser(description='Predict on test set using trained model')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Model checkpoint path (directory containing model files)')
    
    parser.add_argument('--test_csv', type=str, default='test_data.csv',
                        help='Test CSV file path')
    parser.add_argument('--test_img_dir', type=str, default='test_images',
                        help='Test image directory')
    
    parser.add_argument('--output_csv', type=str, default='test_data_predictions.csv',
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

