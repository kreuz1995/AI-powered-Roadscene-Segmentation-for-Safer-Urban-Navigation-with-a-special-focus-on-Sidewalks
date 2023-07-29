import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
from classes import CLASSES
import sys
sys.path.append('../model_f')
from model import *
from torch.utils.data import Dataset as BaseDataset

DATA_DIR = 'Dataset'
x_test_dir = os.path.join(DATA_DIR, 'XTest')
y_test_dir = os.path.join(DATA_DIR, 'yTest')

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=None, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
) 

test_dataloader = DataLoader(test_dataset, num_workers = 3)

metrics = [
    IoU(threshold=0.5),
    Accuracy(threshold=0.5),
    Fscore(threshold=0.5),
    Recall(threshold=0.5),
    Precision(threshold=0.5),
]

dummy = torch.load("Pytorch_Weights/DeepLabV3Plus_resnet101.pth")
Trained_model = dummy

# Evaluate model on test set
test_epoch = ValidEpoch(
    model=Trained_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)