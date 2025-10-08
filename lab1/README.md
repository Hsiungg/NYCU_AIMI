# Lab1 - Chest X-ray Classification

## Overview
This project implements **binary chest X-ray image classification** using deep learning models such as **ResNet-18**, **ResNet-50**, and **Swin V2**.  
The goal is to classify X-ray images as **normal** or **pneumonia**, and evaluate models using **Accuracy**, **Precision**, **Recall**, and **F1-score**.

---

## Dataset
The dataset contains chest X-ray images. To download and prepare the dataset:

1. Access the dataset at: [Download Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract the dataset into the project directory:

```
project_root/
│
├── chest_xray/
│── train.py
│── test.py
```

## Usage

After installing all the dependencies:

### Training
To train a model, run the following command:

```bash
python train.py --model_name {model_name e.g resnet18, resnet50, swin_v2}
```

### Testing
To test a model, run the following command:

```bash
python test.py --model_name {model_name e.g resnet18, resnet50, swin_v2} --model_path {your_path_to_model_weight.pt}
```

