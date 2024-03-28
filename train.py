import cv2
import pandas as pd
import numpy as np 
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.metrics import classification_report, confusion_matrix

import fire
import glob

from data import get_signal_dataset
from redes import cnn_1d_signal_classifier, cnn_1d_manu


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print("Error setting memory growth")
        print(e)

def main():
        
    # Parámetros para los generadores
    params = {
        'name': 'manu_arch',
        'dim': (224, 224, 3),
        'batch_size': 32,
        'name_DB': 'DataSet Coloreado/colored_images',
        'level_rd': ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"],
        'n_classes': 5,
        'shuffle': True
    }

    csv_list = glob.glob('zhang-wamsley-2019/data/CSV/*.csv')

    train_dataset = get_signal_dataset(
        paths_csv=csv_list[:30],
        shuffle=True
    )

    val_dataset = get_signal_dataset(
        paths_csv=csv_list[150:180]
    )
    
    # test_dataset, test_labels = get_dataset(path_csv="csv/test_filtered_images.csv",
    #                             shuffle=False,
    #                             batch_size=params['batch_size'],
    #                             gray_scale=False,
    #                             return_data=True)


    # model = cnn_1d_signal_classifier()
    model = cnn_1d_manu()
    # model = modelo_secuencial
    
    model.summary()

    # sparse_categorical_crossentropy
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(0.0001),
        metrics = ['acc']) 
                    #tfam.F1Score(num_classes=5, name='f1_weig', threshold=0.5)])

    monitoring_metric = 'val_acc'
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor = monitoring_metric, 
            mode = 'max', 
            patience = 40, 
            verbose=1,
            restore_best_weights=False),
        tf.keras.callbacks.ModelCheckpoint(
            filepath = 'models/CNN_Manu_source/model.keras',
            monitor = monitoring_metric,
            mode = 'max',
            save_best_only = True,
            verbose = 1),
        tf.keras.callbacks.TensorBoard(
            os.path.join('logs/', params['name']))
        #tf.keras.callbacks.ReduceLROnPlateau(monitor=monitoring_metric, factor=0.5, patience=5)
        ]

    # Entrenamos el modelo
    history = model.fit(train_dataset.repeat(),
            validation_data=val_dataset,
            #class_weight = class_weight,
            epochs = 50,
            steps_per_epoch=4000,
            callbacks = callbacks)

    # Guardamos el modelo
    model.save("CNN_source.h5")

    # Evaluamos el modelo
    # predictions = model.predict(test_dataset, batch_size = 64)
    #predictions = model.predict(test, batch_size = 64).round()
    
    #Gráficas
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    #plt.savefig('graficas/LossCurves'+cuerpo+numeroEpocas+'DataSetNadia'+'.png')
    plt.savefig('Loss'+'.png')
    plt.show()

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0) #Sera accuracy en lugar de acc si has puesto como metrics del model.fit, accuracy
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    #plt.savefig('graficas/AccuracyCurves'+cuerpo+numeroEpocas+'DataSetNadia'+'.png')
    plt.savefig('Accur'+'.png')
    plt.show()

    # print(confusion_matrix(keras.utils.to_categorical(test_labels, num_classes = params['n_classes']), predictions))    
    # print(classification_report(keras.utils.to_categorical(test_labels, num_classes = params['n_classes']).argmax(axis = 1), predictions.argmax(axis = 1), target_names = params['level_rd']))


if __name__ == "__main__":
    fire.Fire(main)