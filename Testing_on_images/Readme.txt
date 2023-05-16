This folder has the Test_images.py file, which will generate segmented output for an input of RGB images.

Please follow the below steps for execution:

1. Copy your input images to the folder named "Input".
2. Execure the below 2 commands in your command line:
   pip install -U opencv-contrib-python-headless matplotlib numpy albumentations[imgaug] pip setuptools
   pip install git+https://github.com/qubvel/segmentation_models.pytorch

   These 2 commands will install the necessary libraries for the execution of this code.
3. now just execute the Test_images.py using the command "python Test_images.py".

Your output images will be generated and stored in the folder "Output".

Please remember to clean the Output folder to run this code for the 2nd time after running for the 1st time. Otherwise, it might overwrite the existing files.

Note: This code has been developed keeping in mind a Linux system. So, all the folders have a forward slash in their paths in the code. If you plan to execute on Windows, you have to manually change the forward slashes to backward slashesin the code.