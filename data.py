import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import random
from PIL import Image
from tensorflow._api.v2.data import experimental
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE


def num_labels(df):
    d_labels = {'W': 0, '1': 1, '2': 2, '3': 3, '4': 4, 'e': 5, 'R': 6, '?': 7}

    df['label'] = df['label'].map(d_labels)

    return df


def gather_signals_by_class(df_raw, num_signals=20):
    """
    Organize data by class, collecting sets of `num_signals` signals per class.
    """
    grouped = df_raw.groupby('label')
    grouped_data = []
    
    for _, group in grouped:
        for i in range(0, len(group), num_signals):
            subset = group.iloc[i:i+num_signals]
            if len(subset) == num_signals:
                grouped_data.append(subset)
                
    return pd.concat(grouped_data)

def get_signal_dataset(paths_csv: list = None, shuffle: bool = False, batch_size: int = 32, 
                over_samp: bool = False, return_data: bool = False, num_signals: int = 20):
    # Load CSV data
    df_raw = pd.DataFrame()
    
    # Load and concatenate CSV files
    for path in paths_csv:
        temp_df = pd.read_csv(path, delimiter=';', header=0)
        df_raw = pd.concat([df_raw, temp_df], ignore_index=True)    
    
    df_raw = num_labels(df=df_raw)
    
    if over_samp:
        # Implement over_sampling logic here
        pass

    # Organize data by class, collecting sets of num_signals signals per class
    data_processed = gather_signals_by_class(df_raw, num_signals=num_signals)
    
    if shuffle:
        # Shuffling here would disrupt the grouping, consider shuffling within groups if necessary
        pass

    # Extracting sensor data and labels
    sensor_data = data_processed[['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7']].values
    labels = data_processed['label'].values[::num_signals]  # Assuming the same label for each group of 20 signals

    # Reshape data to have 'num_signals' signals per item
    sensor_data_reshaped = sensor_data.reshape(-1, num_signals, sensor_data.shape[1])
    labels_one_hot = tf.keras.utils.to_categorical(labels)

    # Create TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((sensor_data_reshaped, labels_one_hot))

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if return_data:
        return dataset, labels_one_hot
    else:
        return dataset







copy_class = {'0': 3, '1': 3, '2': 3, '3': 3, '4': 3}

def over_sampling(data_raw):
    print('over_sampling')
    data_raw = pd.concat([pd.concat([data_raw[data_raw['level']==int(key)]]*value) for key, value in copy_class.items()]).sample(frac=1)

    return data_raw

def down_sampling(data_raw):
    print('down_sampling')\

    
    return data_raw

def tf_augmenter():
    @tf.function
    def f(*dataset):
        output= list(dataset)
        image = output[0]
        #hacer aleatorio  Y a lo mejor algún random crop
        if tf.random.uniform([1], minval=0, maxval=1) > 0.5:
            image = tf.image.random_flip_up_down(image)
        if tf.random.uniform([1], minval=0, maxval=1) > 0.5:
            image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([1], minval=0, maxval=1) > 0.7:
            image = tf.image.random_brightness(image, 0.15)
        if tf.random.uniform([1], minval=0, maxval=1) > 0.7:
            image = tf.image.random_contrast(image, 0.6, 1.4)
        if tf.random.uniform([1], minval=0, maxval=1) > 0.5:
            image = tf.image.random_jpeg_quality(image, 80, 100)

        output[0] = image
        return output
    return f

def tf_augmenter2():
    @tf.function
    def f(*dataset):
        output= list(dataset)
        image = output[0]
        #hacer aleatorio  Y a lo mejor algún random crop
        image = tf.signal.rfft3d(image)
        #(image, 80, 100)

        output[0] = image
        return output
    return f

@tf.function
def load_image(*inputs):
    outputs = list(inputs)
    image = tf.numpy_function(load_image_np, [inputs[0]], tf.float32)
    image.set_shape([None, None, 3])
    outputs[0] = image
    
    return outputs

def load_image_np(path):
    return np.array(Image.open(path).convert('RGB')).astype(np.float32)
    #return np.array(cv2.imread(path))

def resize(index=0, resize_to=None):
    def f(*dataset):
        output = list(dataset)
        resized_image = tf.image.resize(dataset[index], resize_to)
        resized_image = tf.cast(resized_image, tf.uint8)
        output[index] = resized_image
        
        return output
    return f

def preprocess_input(index):
    @tf.function
    def f(*dataset):
        output = list(dataset)
        image = dataset[index]
        image = tf.cast(image, tf.float32)
        image = image / 255.
        output[index] = image
        
        return output
    return f

def get_image_dataset(
    path_csv: str = None,
    shuffle: bool = False,
    batch_size: int = None,
    gray_scale: bool = False,
    augmenter: bool = False,
    cache: bool = False,
    return_data: bool = False,
    over_samp: bool = False,
    num_aug: int = None,
    )->tf.data.Dataset:

    data_raw = pd.read_csv(path_csv, header = 0).sample(frac=1)
    if over_samp:
        data_raw = over_sampling(data_raw)
    
    #data_raw = down_sampling(data_raw)

    print('Number of instances per label: ',
          pd.Series(data_raw['level']).value_counts(), sep='\n')
    print('Percentaje of instances per label: ',
          pd.Series(data_raw['level']).value_counts().div(pd.Series(data_raw['level']).shape[0]),
          sep='\n')

    names = np.array(data_raw['image'], dtype=str)
    labels = np.array(tf.keras.utils.to_categorical(data_raw['level'], num_classes=5))
    data = names, labels

    dataset = tf.data.Dataset.from_tensor_slices(data)
    #print(list(dataset.as_numpy_iterator()))

    if shuffle:
        print(' > Shuffle')
        dataset = dataset.shuffle(len(names))

    # Loading Images
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Resize a lo que queremos
    #dataset = dataset.map(resize(resize_to=(224,224)), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Cacheteamos el dataset 
    #if cache:
    #    print('cacheteamos el dataset')
    #    dataset = dataset.cache()

    ##############################################################
    # Aumenters
    if augmenter:
        print(f' > Augmentamos datos numero {num_aug}')
        if num_aug == 1:
            dataset = dataset.map(tf_augmenter(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif num_aug == 2:
            dataset = dataset.map(tf_augmenter2(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # preprocesamos las imagenes
    dataset = dataset.map(preprocess_input(0), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if gray_scale:
        print(' > Escala de grises')
        dataset = dataset.map(lambda *args: (tf.image.rgb_to_grayscale(args[0]), *args[1:]))

    # Prepare batch_size
    if batch_size is not None:
        print(' > Establecemos el batchsize')
        dataset = dataset.batch(batch_size)
    
    # Con esto le decimos que vaya prepando el siguiente bacth mientras est'a con el actual
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if return_data:
        return dataset, labels
    else:
        return dataset