from keras.preprocessing.image import ImageDataGenerator
import numpy as np, pandas as pd, sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
def prep():
    generator=ImageDataGenerator(validation_split=0.10)
    train_df=pd.read_csv('../Data/Pose_Dataset/train_joints_coords.csv', header=None)
    test_df=pd.read_csv('../Data/Pose_Dataset/test_joints_coords.csv', header=None)
    train_img='../Data/Pose_Dataset/train/'
    test_img='../Data/Pose_Dataset/test/'
    train=generator.flow_from_dataframe(train_df, directory=train_img, batch_size=40,
                                 target_size=(224, 224), x_col=0,
                                 y_col=list(np.arange(1,15,1)), class_mode= 'raw',
                                 subset="training")
    valid=generator.flow_from_dataframe(train_df, directory=train_img, batch_size=40,
                                        target_size=(224, 224), x_col=0,
                                        y_col=list(np.arange(1,15,1)), class_mode= 'raw',
                                        subset="validation")
    generator=ImageDataGenerator()
    test = generator.flow_from_dataframe(test_df, directory=test_img, x_col=0,
                                         y_col=list(np.arange(1,15,1)), class_mode= 'raw',
                                         target_size= (224,224), batch_size=40)
    print("Reading Data...")
    return train, valid, test

def prep2():
    #Importing Action Joints Dataset
    df = pd.read_csv('../Data/Action_Dataset/action_joints.csv')
    df.columns= list(range(df.shape[1]))
    x=df.iloc[:, 1:-1] 
    y=df.iloc[:, -1].values.reshape(-1,1)
    enc= OneHotEncoder()
    y= enc.fit_transform(y).toarray()
    x_train, x_val, y_train, y_val = train_test_split(x,y, test_size= 0.1) 
    return x_train, y_train, x_val, y_val, enc  
    
    