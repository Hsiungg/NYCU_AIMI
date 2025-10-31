# Lab2 - EEG Classification with BCI competition dataset

## Overview

In this lab, our task is **EEG classification using the BCI competition dataset**, with **EEGNet** as the primary model. We implemented the EEGNet architecture and **visualized** the training process, while experimenting with different parameter settings to improve classification accuracy. In addition, we implemented the **DeepConvNet** model to further enhance performance. In the following sections, I will first introduce EEGNet.

---

## Dataset

The dataset contains chest X-ray images. To download and prepare the dataset:

1. Extract the dataset into the project directory using command below:

```bash
unzip lab2_EEG_classification.zip 'lab2_EEG_classification/*' -d temp && mv temp/lab2_EEG_classification/* . && rm -r temp
```

1. And the project directory structure should be look like this,

```text
project_root/
│
├── data/
|   |── S4b_test.npz
|   |── S4b_train.npz
|   |── X11b_test.npz
|   |── X11b_train.npz
│── main.py
│── eval.py
│── plotting.py
│── util.py
```

## Usage

After installing all the dependencies using requirements.txt via pip:

```bash
python -m pip install -r requirements.txt
```

### Training

To train a model, run the following command:

```bash
python main.py --model {EEGNet, DeepConvNet}
```

you can also add -grid_search for using grid search during training.

### Testing

To test a model, run the following command:

```bash
python eval.py --model {EEGNet, DeepConvNet} --model_path {your_path_to_model_weight.pt}
```