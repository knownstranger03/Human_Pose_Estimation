#Import all necessary Packages and Functions
import os, random, cv2, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from numpy import*


#ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore")
#Defining a new function
def prep():
    #Load Explanatory Data (Images)
    train_dir='../Data/Pose_Dataset/train' #set train images path
    test_dir='../Data/Pose_Dataset/test' #set test images path
    #list all files in the specified directories
    trlist, telist = os.listdir(train_dir), os.listdir(test_dir)
    #Load all the images and flatten as numpy arrays
    trmatrix = array([array(cv2.imread('../Data/Pose_Dataset/train'+'//'+im2)).flatten()
                   for im2 in trlist], 'f')
    tematrix = array([array(cv2.imread('../Data/Pose_Dataset/test'+'//'+im2)).flatten()
                   for im2 in telist], 'f') 
    #Load Join_coordinates (target (Y) variables)
    train_jd=pd.read_csv('../Data/Pose_Dataset/train_joints_coords.csv',
                          names=['img_file_name',1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    test_jd=pd.read_csv('../Data/Pose_Dataset/test_joints_coords.csv',
                         names=['img_file_name', 1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    #Remove the Image_file_names: must not be fed into the model
    train_jd1=train_jd.drop('img_file_name', axis=1)
    test_jd1=test_jd.drop('img_file_name', axis=1)
    #Train data
    x_train,y_train=(trmatrix, train_jd1)
    x_train = x_train.reshape(x_train.shape[0], 224, 224, 3)
    x_train = x_train.astype('float32')
    #Test data
    x_test, y_test = tematrix, test_jd1
    x_test = x_test.reshape(x_test.shape[0], 224, 224, 3)
    x_test = x_test.astype('float32')
    
    print('Seq === x_train, y_train, x_test, y_test')
    return x_train, y_train, x_test, y_test












