import os
import torch
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.nn import DataParallel
import sys
sys.path.append('..')
from classes import *
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """TrayDataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes,
            augmentation=None, 
            preprocessing=None,
    ):
        #get images(x) and masks(y) ids
        self.ids_x = sorted(os.listdir(images_dir))
        #['1001a01.jpg', '1005a.jpg', '1006a72.jpg', '2001a72.jpg', '2002a.jpg'] etc.
        self.ids_y = sorted(os.listdir(masks_dir))
        
        #get images(x) and masks(y) full paths (fps)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_x]
        #'/content/drive/My Drive/Colab Notebooks/TrayDataset/XTest/1001a01.jpg'
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_y]
        
        # convert str names to class values on masks
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i],0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # mask = np.stack(masks, axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
 
        return image, mask
        
    def __len__(self):
        return len(self.ids_x)

import albumentations as albu
def get_training_augmentation():
    train_transform = [

        albu.Resize(256, 320, p=1),
        albu.HorizontalFlip(p=0.5),

        albu.OneOf([
            albu.RandomBrightnessContrast(
                  brightness_limit=0.4, contrast_limit=0.4, p=1),
            albu.CLAHE(p=1),
            albu.HueSaturationValue(p=1)
            ],
            p=0.9,
        ),

        albu.IAAAdditiveGaussianNoise(p=0.2),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(256, 320)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import segmentation_models_pytorch as smp
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax' 
model =smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION)
#Normalize your data the same way as during encoder weight pre-training
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
Trained_model=torch.load("../Pytorch_Weights/DeepLabv3Plus_resnet101.pth",map_location=DEVICE)

'''-----------------------------function to segment-------------------------------'''
import csv

def decode_segmentation_map(image, classesLength=66):
    Class_label_colors = {}
    
    with open('../class_colors.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_label = int(row['class_label'])
            red = int(row['red'])
            green = int(row['green'])
            blue = int(row['blue'])
            Class_label_colors[class_label] = (red, green, blue)

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, classesLength):
        idx = image == l
        r[idx] = Class_label_colors[l][0]
        g[idx] = Class_label_colors[l][1]
        b[idx] = Class_label_colors[l][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

#Code that will generate the output in the output folder

from PIL import Image
import time

frames_folder='Input'
folder_path='Input'
# os.makedirs("Final_Folder")
output_path='Output'
#To check the iteration
start_time = time.time()
num_iterations = len(folder_path)
update_interval = 1

x_test_dir = frames_folder
y_test_dir = frames_folder
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir,
    augmentation=None, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
) 


for j in range(len(test_dataset)):
    image,gt_mask = test_dataset[j]
    # print(image)
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    predicted_mask = Trained_model.module.predict(x_tensor)
    predicted_output = torch.argmax(predicted_mask.squeeze(), dim=0).detach().cpu().numpy()
    rgb_map = decode_segmentation_map(predicted_output,65)
    fig, ax = plt.subplots()
    # ax.imshow(image)
    ax.imshow(rgb_map,alpha=0.8)
    ax.axis('off')
    image_name = os.path.basename(test_dataset.images_fps[j])
    output_name = os.path.splitext(image_name)[0] + '.png'
    fig.savefig(os.path.join(output_path, output_name), bbox_inches='tight')
    plt.close(fig)
    print('\rImage Segmented:', j+1, end='')
print("\nOutput generated for all the images")