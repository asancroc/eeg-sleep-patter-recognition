import fire
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from sklearn.metrics import confusion_matrix
from data import get_tf_dataset
from sklearn.metrics import confusion_matrix
from exp_utils import (
    display_roc_and_f1, parse_experiment
)


@parse_experiment
def evaluate(
    experiment_name: str,
    csv_pattern: str,
    l_users_test: list,
    window_size: int = 20,
    batch_size: int = 32,
    normalization: str = None,
    overlap: int = 5,
    pca: bool = False,
    num_features: int = 63,
    **experiment,
):

    # Load the trained model
    model = load_model(os.path.join('models', experiment_name, 'model.h5'), compile=False)

    # Load the test set
    csv_list = glob.glob(csv_pattern)

    test_dataset, labels = get_tf_dataset(
        paths_csv=csv_list,
        window_size=window_size,
        batch_size=batch_size,
        shuffle=False,
        l_users = l_users_test,
        normalization=normalization,
        window_aug=False,
        overlap=overlap,
        num_features=num_features,
        pca=pca,
    )

    y_pred_scores = model.predict(test_dataset)

    y_pred_scores_bin = np.where(y_pred_scores >= 0.5, 1, 0)
    cf_matrix = confusion_matrix(labels, y_pred_scores_bin)
    print(cf_matrix)
    
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
    plt.show()
    
    display_roc_and_f1(labels, y_pred_scores)


if __name__ == "__main__":
    # fire.Fire(evaluate)
    fire.Fire(evaluate(experiment = 'prueba_final'))