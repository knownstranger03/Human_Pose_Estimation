import tensorflow
def coeff_determination(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

def optimizer():
    from tensorflow.keras.optimizers import Adam
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    return adam
