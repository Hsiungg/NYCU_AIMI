# Lab3 - Multi-class Classification with CXR dataset

## Overview

The task is **multi-class image classification on chest X-ray images**. The goal is to accurately predict the disease category for each image, despite challenges such as class imbalance, subtle visual differences, and varying image quality.

The dataset contains multiple disease categories and is pre-split into train, validation, and test sets, each maintaining roughly the same class distribution. Images are preprocessed with resizing, normalization, and augmentation to standardize input for the models.

Model performance is evaluated primarily using the **Macro F1-score**, which ensures balanced assessment across all classes. Accuracy and per-class F1 are also reported for additional analysis.

---

## Dataset

The dataset is provide by Kaggle competition which contains chest X-ray images. To download and prepare the dataset:

1. Download using Kaggle CLI:

```bash
kaggle competitions download -c cxr-multi-label-classification
```

2. And Extact using:
```bash
unzip cxr-multi-label-classification.zip
```

And the project directory structure should be look like this,

```text
project_root/
│
├── test_images/
├── train_images/
├── val_images/
├── train_data.csv
├── val_ddata.csv
│── train.py
│── predict.py
```

## Usage

### Training

To train a model, run the following command:

```bash
python train.py --model_type {vit, swin, timm, dinov2, dinov3, eva02} --num_epochs 100 --batch_size 8 --learning_rate 5e-5
```

**Available model types:**
- `vit`: Vision Transformer
- `swin`: Swin Transformer
- `timm`: TIMM models (default)
- `dinov2`: DINOv2
- `dinov3`: DINOv3
- `eva02`: EVA-02

**Optional arguments:**
- `--model_name`: Specific pretrained model name (e.g., `microsoft/beit-base-patch16-224`)
- `--output_dir`: Output directory for checkpoints (default: `./output/hf_model_{model_type}`)
- `--use_focal_loss`: Use Focal Loss instead of Cross Entropy Loss (default: True)
- `--focal_gamma`: Gamma parameter for Focal Loss (default: 2.0)
- `--no_resample`: Disable resampling for class imbalance
- `--no_class_weights`: Disable class weights

**Example:**
```bash
python train.py --model_type timm --model_name microsoft/beit-base-patch16-224 --num_epochs 50 --batch_size 8
```

### Testing

To test a model, run the following command:

```bash
python predict.py --checkpoint_path {path_to_model_checkpoint}
```