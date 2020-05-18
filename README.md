# Human_Pose_Estimation
This repository contains code for human pose keypoints estimation and action classification.

NOTE: For evaluation kindly look at the Data_Visualization.ipynb

# Installation

1. Open CMD or Terminal window and install git
2. Execute this command to clone the repo git clone https://github.com/knownstranger03/Human_Pose_Estimation
3. To get the backend Custom models contact me on sidhuprasad0304@gmail.com for a drive link as github limits the file size        for upload
4. cd into the cloned folder and open Data_Visualization.ipynb from jupyter notebook and predict with custom image inputs.
####You can change any functions and model architectures as required.



# Contents
1. Custom_Functions: this folder contains custom functions built for internal data pipelines and model architecture builders.
2. Custom_Models: this folder contains both trained and untrained models (architecture) created for both prediction of pose keypoints and classification of human action.
3. Data: contains two sub folders: -Action_Dataset -Pose_Dataset
    - Action_Dataset: contains Action_Joints.csv with coordinates and labels Hello/Namaste
    - An mp4 video which is used in this project for final testing and visualization
    - Images folder that has 23 images for training and testing the classification
    
    - Pose_Dataset: Contains Train and test coordinates for human pose keypoint detection/estimation
    - Two folders named Train//Test, which contain the preprocessed images from FLIC Dataset.
    
4. Data_Visualization.ipynb : This notebooks presents the results of both the final models and their visualization on the final video.
5. Model_Trainnig.ipynb: This notebook was used to train both the final Keypoint detection and classification models on Google Colab with GPU.
6. run.py this file can be used a module to import two functions respectively -pred -classify
    -pred : Is a function that requires image input whose ndim=[244,244,3] and return human pose keypoints predicted on the           image. This function can directly be used by anyone to retrive 14keypoints of 7joints, it supports single image/multiple       images/image pipelines input types.
    -classify : This function takes arrays of coordinates of joints as input and returns a label Hello/Namste. Supports               multiple array inputs for various image coordinates as a list of arrays.
    
# SCOPE

** This repo will be updated with 2 .ipynb files with functions that can includes a flask based web application which allows user to directly predict and view results in a .html file. The new update will also include a feature to accept prediction inputs directly from the user's webcam.

** With human pose estimation detecting crimes such as physical assault, robery, etc can be achieved, thereafter used in realtime to alert the authorities.

# Additional Details
1. Must use tensorflow 2.2.0
2. If error while loading the model use:
import tensorflow as tf
from Metrics import coeff_determination #this module is available under Custom_Functions folder
tf.keras.models.load_model('path/model.h5', custom_objects={'coeff_determination'=coeff_determination})

# Contributors
Any interested contributors to work on developing Human Pose Detection as a crime detection model contact me on:
 | sidhuprasad0304@gmail.com \\ www.linkedin.com/krishnaprasad03 |
