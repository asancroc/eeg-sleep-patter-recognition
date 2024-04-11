from keras.models import Model
import tensorflow as tf
from keras.layers import (
    Input, Conv1D, GRU, LSTM,
    MaxPooling1D, Flatten, Dense,
    Dropout, SeparableConv1D, AveragePooling1D,
    GlobalAveragePooling1D, BatchNormalization,
    SeparableConv2D, Dropout, AveragePooling2D,
    GlobalAveragePooling2D, BatchNormalization,
)

def cnn_2d_manu(input_shape=(45, 45, 63)):
    x = inp = Input(shape=input_shape)

    # 1st Separable Depthwise 2D Convolution Block
    x = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(x)  # Kernel size inferred from output shape change
    x = Dropout(0.2)(x)  # Assuming dropout rate of 0.2 as an example

    # 2nd Separable Depthwise 2D Convolution Block
    x = SeparableConv2D(64, (3, 3), padding='valid', activation='relu')(x)  # Kernel size inferred from output shape change
    x = Dropout(0.2)(x)  # Assuming dropout rate of 0.2 as an example

    # Average Pooling 2D
    x = AveragePooling2D(pool_size=(2, 2))(x)  # Pool size inferred from output shape change

    # Global Average Pooling 2D
    x = GlobalAveragePooling2D()(x)

    # Batch Normalization
    x = BatchNormalization()(x)

    # Classifier
    x = Dense(1, activation='sigmoid')(x) 

    return Model(inputs=inp, outputs=x)
  
def cnn_1d_manu(num_signals: int, num_features: int):
    x = inp = Input(shape=(num_signals, num_features))  # Adjusted for signal input shape

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

    

def LSTMGRUnn(num_signals: int,  num_features: int):
    
    x = inp = Input(shape=(num_signals, num_features))

    # Base Model
    x = LSTM(30, dropout=0.2,recurrent_dropout=0.5, activation="tanh",
                 return_sequences=True, return_state=False, stateful=False)(x)
    x = GRU(units=50,dropout=0.2,recurrent_dropout=0.5, return_sequences=True, activation="tanh")(x)
    x = GRU(units=50, dropout=0.2,recurrent_dropout=0.5,return_sequences=False, activation='tanh')(x)

    # Clasificador
    x = Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(inp, x)

def cnn_1d_signal_classifier(num_signals: int, num_features: int):
    x = inp = Input(shape=(num_signals, num_features))  # Adjusted for signal input shape

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