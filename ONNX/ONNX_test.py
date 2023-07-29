import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.nn import DataParallel
import torch
from sklearn.model_selection import train_test_split
import torch
import onnxruntime as ort
from PIL import Image
import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_directory)
from classes import CLASSES

from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """TrayDataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
    
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes,
    ):
        # Get images(x) and masks(y) ids
        self.ids_x = sorted(os.listdir(images_dir))
        self.ids_y = sorted(os.listdir(masks_dir))
        
        # Get images(x) and masks(y) full paths (fps)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_x]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_y]
        
        # Convert str names to class values on masks
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]
    
    def __getitem__(self, i):
        
        # Read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        image = cv2.resize(image, (544, 544))
        mask = cv2.resize(mask, (544, 544))
        
        # Extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        return image, mask
        
    def __len__(self):
        return len(self.ids_x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import segmentation_models_pytorch as smp
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax' 
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

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


# Load ONNX model
onnx_model_path = "ONNX_Weights/DeepLabV3Plus_resnet101.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Output directory for segmented images
output_path = 'Output'

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

frames_folder = 'Input'
folder_path = 'Input'
num_iterations = len(folder_path)
update_interval = 1

x_test_dir = frames_folder
y_test_dir = frames_folder
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir,
    classes=CLASSES,
)

for j in range(len(test_dataset)):
    image, gt_mask = test_dataset[j]

    # Convert input image to float32
    image = image.astype(np.float32)
    image = preprocessing_fn(image)

    x_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Run the ONNX model for segmentation
    ort_inputs = {ort_session.get_inputs()[0].name: x_tensor.detach().cpu().numpy().astype(np.float32)}
    ort_outputs = ort_session.run(None, ort_inputs)
    predicted_output = np.argmax(ort_outputs[0][0], axis=0)
    rgb_map = decode_segmentation_map(predicted_output, 65)

    fig, ax = plt.subplots()
    ax.imshow(rgb_map, alpha=0.8)
    ax.axis('off')

    image_name = os.path.basename(test_dataset.images_fps[j])
    output_name = os.path.splitext(image_name)[0] + '.png'
    fig.savefig(os.path.join(output_path, output_name), bbox_inches='tight')
    plt.close(fig)
    print('\rImage Segmented:', j+1, end='')

print("\nAll Images segmented")