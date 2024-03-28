from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, SeparableConv1D, AveragePooling1D, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf

def cnn_1d_signal_classifier():
    x = inp = Input(shape=(20, 63))  # Adjusted for signal input shape

    # 1st Convolutional Block
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Top Model
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    # Classifier
    x = Dense(1, activation='sigmoid')(x)  # Adjusted for 8 classes

    return Model(inputs=inp, outputs=x)
  
  
def cnn_1d_manu():
    x = inp = Input(shape=(20, 63))  # Adjusted for signal input shape

    # 1st Convolutional Block
    x = SeparableConv1D(64, 8, activation='relu')(x)
    x = SeparableConv1D(64, 8, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = AveragePooling1D(pool_size=2)(x)
    x = GlobalAveragePooling1D()(x)
    
    
    x = BatchNormalization()(x)

    # Classifier
    x = Dense(1, activation='sigmoid')(x)  # Adjusted for 8 classes

    return Model(inputs=inp, outputs=x)