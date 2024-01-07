# A Project on Roadscene Segmentation with a special focus on Sidewalks (Class 16).
This repository contains code to generate segmented images and segmented videos of roadscenes, with a special focus on sidewalks. The four models that we have used here are:
1. DeeplabV3Plus. 2. UNet++. 3. PSPNet. 4. PAN
All of them have been trained on resnet101 as encoder and imagenet as the pretrained weights. Then they were first trained on Cityscapes dataset (https://www.cityscapes-dataset.com/file-handling/?packageID=3) and then finetuned on a special subset of the Mapillary Vistas dataset, with Sidewalks on focus.

Link to the completed project report: https://drive.google.com/file/d/11XpDHQ9ny6pVxRzJWPyHuSW0JG5vsj5S/view?usp=sharing   
Link to the pretrained-weights: https://drive.google.com/drive/folders/1U6un3tbRwSO1XFbyZwbbU92EoiD5QpC_?usp=sharing

Below are the details of all the files and specific instruction to execute them.

## This repository contains the below files:
1. train_pipeline.py: Python script which needs to be executed to train the model with modified input.
2. test.py: Provided the inference of the model.
3. requirements.txt: Contains all the necessary libraries.
4. model.py: Contains the dataset class and the declaration of the model. The model architecture can be changed by changing the encoder, decoder, and pre-trained weights in the code.
5. classes.py: Contains the list of the classes. For training, please declare all the classes in a list in this script.
6. class_colors.csv: This CSV file contains all the RGB values of the classes with class names. Changing the pixel value here will effect that class in all the codes. This is a singlehanded file that is connected to all the Training and testing codes. So, changing the value here affects all of them.

## Training:
1. pip install -r requirements.txt
2. Copy all of your training and validation images in the Dataset folder in XTrain, yTrain etc. X-prefixed folders will contain the real images, and y-prefixed folders will contain the annotated images.
3. To use a pretrained weights, please make required changes to the pretrain_weights comment in model.py
4. Make required changes to the classes.py and class_colors.csv.
5. Run train_pipeline.py
6. The new trained weights are saved in Pytorch_Weights folder.

## Inference: 
1. change the name of your pretrained weight to the newly trained weight in the test.py script.
2. run test.py

## ONNX
1. Code to both conversion to onxx and testing on ONNX are given in the ONNX folder.
2. It will take weights from the Pytorch_Weights folder in the main directory.
3. Just run the ONNX_Export.py file to convert a pytorch weight to ONNX format.
4. And to generate segmented images using that weight, use the ONNX_test.py

## Testing on images
1. Go to Testing_on_images folder.
2. Copy your input dataset to the "Input" Folder.
3. Run Test_images.py.
4. The images will be segmented and be generated in Output folder.

## Testing on Videos:
1. Go to Testing_on_videos folder.
2. Copy your input videos to Final_Folder/Input_Video.
3. This code generated 1 segmented video at a time. Write the name of the particular video file in "input_file_name.txt".
4. Execute Test_on_videos.py.
5. All the segmented videos will be generated in Final_Folder/Output_Video.

# Results:
## 1. On Images:
![image](https://github.com/kreuz1995/SemanticSegmentationSidewalk/assets/106822147/06bd3215-5ee6-4b35-994e-effed73d10b6)

### Metrics: 
![image](https://github.com/kreuz1995/SemanticSegmentationSidewalk/assets/106822147/cb044a01-317c-43cd-930a-f4e61a25aa10)

## 2. On Videos:
### Metrics:
![image](https://github.com/kreuz1995/SemanticSegmentationSidewalk/assets/106822147/254414ba-93cd-45e0-89a9-9d1f1541383b)

