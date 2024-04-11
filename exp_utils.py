import pickle
import numpy as np
import seaborn as sns
import tensorflow as tf
import experiments

import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, auc, confusion_matrix,
    roc_curve,
)

tf.random.set_seed(33)

d_users_split = {
    'train': [
        '249', '266', '010', '388', '088',
        '185', '422', '107', '026', '455',
        '348', '337', '309', '245', '183',
        '411', '374', '384',
    ],
    'val': [
        '042', '347', '512', '228', '452',
    ],
    'test': [
        '083', '126',  '030', '181', '535',
    ],
}



def parse_experiment(func):
    def decorator(*args, **kwards):
        experiment = {}
        if "experiment" in kwards:
            experiment = getattr(experiments, kwards["experiment"])
            del kwards["experiment"]
            kwards.update(experiment)
            func(*args, **kwards)
        else:
            func(*args, **kwards)

    return decorator



def save_ml_model(ml_model, model_path):
    with open(model_path,'wb') as f:
        pickle.dump(ml_model,f)



def load_ml_model(model_path):
    with open(model_path,'rb') as f:
        model = pickle.load(f)
    return model



def normalize_df_columns_1_1(df):
    # Columns to normalize (excluding 'label')
    columns_to_normalize = df.columns[:-1]

    for column in columns_to_normalize:
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = ((df[column] - min_value) / (max_value - min_value)) * 2 - 1
    
    return df



def normalize_df_columns_0_1(df):
    # Columns to normalize (excluding 'label')
    columns_to_normalize = df.columns[:-1]

    for column in columns_to_normalize:
        min_value = df[column].min()
        max_value = df[column].max()
        df[column] = (df[column] - min_value) / (max_value - min_value)
    
    return df



def display_roc_and_f1(labels, y_pred_scores):
    f1 = f1_score(labels, y_pred_scores, average='weighted')
    print(f'F1 Score: {f1}')

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



def display_conf_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    sns.heatmap(conf_matrix/np.sum(conf_matrix), annot=True, 
        fmt='.2%', cmap='Blues')
    plt.show()
