# Evaluamos el modelo
# predictions = model.predict(test_dataset, batch_size = 64)
#predictions = model.predict(test, batch_size = 64).round()

# print(confusion_matrix(keras.utils.to_categorical(test_labels, num_classes = params['n_classes']), predictions))    
# print(classification_report(keras.utils.to_categorical(test_labels, num_classes = params['n_classes']).argmax(axis = 1), predictions.argmax(axis = 1), target_names = params['level_rd']))

import fire
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from keras.models import load_model
import pandas as pd
from sklearn.metrics import confusion_matrix
from config import parse_experiment
from data import get_tf_dataset
from sklearn.metrics import (
    f1_score, confusion_matrix, auc,
    roc_curve, RocCurveDisplay
)


def display_roc(labels, y_pred_scores, f1):

    fpr, tpr, _ = roc_curve(labels, y_pred_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="Curva ROC (área = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR", fontsize=16)
    plt.ylabel("TPR", fontsize=16)
    plt.xticks()
    plt.yticks()
    plt.title("F1 Score: %0.4f" % f1, fontsize=18)
    plt.legend(loc="lower right")
    plt.show()


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
        return_data=True,
        overlap=overlap,
        num_features=num_features,
        pca=pca,
    )

    # Make predictions
    # Replace this with your actual prediction code if your model does not output binary predictions
    y_pred_scores = model.predict(test_dataset)  # This might output continuous scores depending on your model
    # y_pred = (y_pred_scores > 0.5).astype(int)  # Apply threshold to get binary predictions
    
    # Calculate far, frr, and EER
    y_pred_scores_bin = np.where(y_pred_scores >= 0.5, 1, 0)
    f1 = f1_score(labels, y_pred_scores_bin, average='weighted')
    print(f'F1 Score: {f1}')


    cf_matrix = confusion_matrix(labels, y_pred_scores_bin)
    print(cf_matrix)

    # sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    # plt.show()
    
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
    plt.show()
    
    display_roc(labels, y_pred_scores, f1)
    print('Hola')



if __name__ == "__main__":
    # fire.Fire(evaluate)
    fire.Fire(evaluate(experiment = 'f1_mia_lstm_normalize_0_1_PCA_20_bs_32_lr_5_e4_'))