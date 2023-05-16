Welcome to the testing section of the segmentation code. 

This folder contains another folder named Final_Folder, a .pth file named "DeeplabV3Plus_resnet50.pth", which is basically the weights, a Test_on_videos.py file , a text file named "input_file_name.txt", and this readme file.

The Final_Folder contains two folders: Input_Video and Output Video.


Please follow the below sequence to segment a video:

1. Copy your input video in the "Input_Video" folder.

2.open the input_file_name.txt and there, within the single quotes, just write the name of the file. Not the path, just       the name of the video file.

3. Please execute the below 2 lines in command line. It will install the libraries:
   pip install -U opencv-contrib-python-headless matplotlib numpy albumentations[imgaug] pip setuptools
   pip install git+https://github.com/qubvel/segmentation_models.pytorch

4. Please execute "Test_on_videos.py" using python Test_on_videos.py. 

5. After that, just open the "Output_Video" folder in the Final_Folder and you will have your segmented video in there.


Important note: Since, I have worked on  linux system, all the folder paths have forward slashes in the main code. If you are using a Windows operating system, please change them to backward slashes in the code.

**Download the weight from: https://easyupload.io/tazxug
and place in this folder.


Also keep in mind, it is possible sometimes that the code runs halfway through and then crashes. So, some temporary folders will be created, but might not get deleted because the execution was not complete. In such a scenario, please make sure to delete all the folders other then "Input_Video" and "Output_Video", before running the code again.
