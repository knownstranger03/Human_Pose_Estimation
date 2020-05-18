import tensorflow, os
os.chdir('Custom_Functions/')
model1=tensorflow.keras.models.load_model('../Custom_Models/Keras_Model_H5/final_model.h5')
model2=tensorflow.keras.models.load_model('../Custom_Models/Keras_Model_H5/final_classification_model.h5')
def pred(img=None):
    out=model1.predict(img)
    return out
def classify(coords=None):
    out=model2.predict(coords)
    return out












