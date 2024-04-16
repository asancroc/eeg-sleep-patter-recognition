import os
import fire
import glob
import tensorflow as tf
import keras
import tensorflow_addons as tfa

from data import get_tf_dataset
from exp_utils import parse_experiment, calculate_class_weights

def get_callbacks(
    patiente,
    monitoring_metric,
    experiment_name,
):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor = monitoring_metric, 
            mode = 'max', 
            patience = patiente, 
            verbose=1,
            restore_best_weights=False),
        keras.callbacks.ModelCheckpoint(
            filepath = f'models/{experiment_name}/model.h5',
            monitor = monitoring_metric,
            mode = 'max',
            save_best_only = True,
            verbose = 1),
        keras.callbacks.TensorBoard(
            os.path.join('logs/', experiment_name))
        ]

    return callbacks


@parse_experiment
def train(
    experiment_name: str,
    csv_pattern: str,
    l_users_train: list,
    l_users_val: list,
    lr: float = 0.0001,
    window_size: int = 20,
    num_features: int = 63,
    batch_size: int = 32,
    epochs: int = 100,
    steps_per_epoch: int = 4_000,
    patiente: int = 20,
    class_weights: bool = True,
    normalization: str = None,
    window_aug: bool = False,
    overlap: int = 5,
    pca: bool = False,
    **experiment,
):
    # Generamos la lista de csv para el entrenamiento
    csv_list = glob.glob(csv_pattern)

    
    # Creamos los splits para train y val
    train_dataset, labels_train = get_tf_dataset(
        paths_csv=csv_list,
        window_size=window_size,
        batch_size=batch_size,
        shuffle=True,
        l_users = l_users_train,
        normalization=normalization,
        window_aug=window_aug,
        overlap=overlap,
        num_features=num_features,
        pca=pca,
    )

    if class_weights:
        cw = calculate_class_weights(labels_train)

    val_dataset, _ = get_tf_dataset(
        paths_csv=csv_list,
        window_size=window_size,
        batch_size=batch_size,
        shuffle=False,
        l_users = l_users_val,
        normalization=normalization,
        window_aug=window_aug,
        overlap=overlap,
        num_features=num_features,
        pca=pca,
    )

    # Cargamos las arquitecturas y compilamos el modelos
    model = experiment['arq']
    model.summary()

    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(lr),
        metrics = ['acc',
        tfa.metrics.F1Score(num_classes=1, name='f1_weig', threshold=0.5, average='weighted')]
    ) 
    

    # Getting callbacks
    callbacks = get_callbacks(
        patiente=patiente,
        monitoring_metric=experiment['monitoring_metric'],
        experiment_name=experiment_name
    )
    

    # Entrenamos el modelo
    _ = model.fit(train_dataset.repeat(),
            validation_data=val_dataset,
            class_weight = cw,
            epochs = epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks = callbacks
    )

if __name__ == "__main__":
    #fire.Fire(train)
    fire.Fire(train(experiment = 'prueba_final'))