This is the folder that contains the file for training a new model.

There are 6 folders in this location. 
1. XTrain: It contains the real images for training.
2. yTrain: It contains the annotated images for training.
3. XVal: It contains the real images for validation.
4. yVal: It contains the annotated images for validation.
5. XTest: It contains the real images for testing.
6. yTest: It contains the annotated images for testing.

Note, please understand you can change the different parameters of training like the decoder, encoder, optimizer etc.. For that please refer to the code. Here, I am only explaining the steps to execute the python file.

Steps:
1. Copy the images to the proper folder according to the above descriptions.
2. Once done, please execute the below two commands to install the required libraries:
   pip install -U opencv-contrib-python-headless matplotlib numpy albumentations[imgaug] pip setuptools
   pip install git+https://github.com/qubvel/segmentation_models.pytorch
3. Once libraries are also installed, execute the file "Train.py" by python Train.py.
4. The training will start and save a model named "Model_weights.pth". To change this name, please refer to line no. 864 in the python file.
5. Once training is done, to test the trained model, please refer to line no. 875 in the Train.py to understand how to test the model for metrics.

