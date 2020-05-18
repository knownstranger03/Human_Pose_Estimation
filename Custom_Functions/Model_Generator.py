import keras
import tensorflow 
from keras.applications.vgg16 import VGG16
from keras.engine.sequential import Sequential    
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, InputLayer, Conv2D, MaxPool2D, Activation, Concatenate,add
from keras.models import Model
import warnings
warnings.filterwarnings('ignore')

#Define a function that returns final model
def run():
    #Download the VGG16 base model
    conv_base = VGG16(weights= 'imagenet', include_top= False, input_shape= (224,224,3))
    #changing the base model layer to non-trainable, to keep the previous trained layers in tact
    for layer in conv_base.layers:
        layer.trainable= False
    #Creating additional architecture
    def top_model():
        top_model = Sequential()  
        top_model.add(Conv2D(64,(3,3), activation='relu', padding = 'same', 
                             input_shape=conv_base.output_shape[1:])) 
        top_model.add(BatchNormalization())
        top_model.add(MaxPool2D(pool_size=(2,2), strides=(1,1))) 
        top_model.add(Flatten()) 
        top_model.add(Dense(4096, activation='relu')) 
        top_model.add(BatchNormalization()) 
        top_model.add(Dropout(0.5))
        top_model.add(Dense(14//ns, activation='relu'))  #for ns =2 it will be 14//2 == 7
        # Creating a final model based on VGG16 and additional architecture
        model = Sequential() 
        for layer in conv_base.layers:
            model.add(layer)     
        model.add(top_model) 
        return model

    def create_model(n):
        outputs=[]
        for i in range(1,n+1):
            globals()[f'model_{i}'] = top_model()
            outputs.append(globals()[f'model_{i}'].output)
        merged= add(outputs)                              
        output= Dense(14, activation='relu', kernel_initializer= 'Ones')(merged) 
        final_model = Model(inputs= conv_base.input, output= output)
        return final_model
    ns=2 
    model = create_model(ns)
    #Save a copy and freshly import the model
    model.save('../Custom_Models/Keras_Model_H5/Untrained_Model.h5')
    print("Model is saved to '/Untrained_Model.h5'")
    import tensorflow
    model=tensorflow.keras.models.load_model('../Custom_Models/Keras_Model_H5/Untrained_Model.h5')
    return(model) #Returns the model directly to current instance

def run2():
    #Building a Deep Neural Network based classification model
    model=Sequential()
    model.add(Dense(164, input_shape=[14], activation= 'relu', kernel_regularizer='l2', kernel_initializer='TruncatedNormal'))
    model.add(Dense(164, activation='relu'))
    model.add(Dense(546*2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(14, activation= 'relu', kernel_regularizer='l2'))
    model.add(BatchNormalization()) 
    model.add(Dense(2, activation= 'sigmoid'))
    #Save a copy of the untrained model
    model.save('../Custom_Models/Keras_Model_H5/Untrained_Classification_Model.h5')
    print("Model is saved to '/Untrained_Classification_Model.h5'")
    import tensorflow
    model=tensorflow.keras.models.load_model('../Custom_Models/Keras_Model_H5/Untrained_Classification_Model.h5')
    return model