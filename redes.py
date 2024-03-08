from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

def cnn_1d_signal_classifier():
    x = inp = Input(shape=(20, 7))  # Adjusted for signal input shape

    # 1st Convolutional Block
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = Conv1D(32, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # 2nd Convolutional Block
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # Top Model
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Classifier
    x = Dense(8, activation='softmax')(x)  # Adjusted for 8 classes

    return Model(inputs=inp, outputs=x)