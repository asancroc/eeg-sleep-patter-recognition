from redes import *
from exp_utils import d_users_split


prueba_manu_no_normalize_10 = {
    'experiment_name': 'prueba_manu_no_normalize_10',
    'num_signals': 20,
    'arq': cnn_1d_manu(num_signals=20, num_features=63),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['train'],
    'l_users_val': d_users_split['train'],
    'batch_size': 64,
    'epochs': 1_000,
    'steps_per_epoch': 1_000,
    'lr': 0.0001,
    'patiente': 20,
    'monitoring_metric': 'val_acc',
}

prueba_manu_normalize_1_1 = {
    'experiment_name': 'prueba_manu_normalize_1_1',
    'num_signals': 50,
    'arq': cnn_1d_manu(num_signals=50, num_features=63),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['train'],
    'l_users_val': d_users_split['val'],
    'batch_size': 32,
    'epochs': 1_000,
    'steps_per_epoch': 1_000,
    'lr': 0.0005,
    'patiente': 25,
    'monitoring_metric': 'val_acc',
    'normalization': '[-1,1]',
}

prueba_manu_normalize_0_1 = {
    'experiment_name': 'prueba_manu_normalize_0_1',
    'num_signals': 100,
    'arq': cnn_1d_manu(num_signals=100, num_features=63),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['train'],
    'l_users_val': d_users_split['train'],
    'batch_size': 512,
    'epochs': 1_000,
    'steps_per_epoch': 1_000,
    'lr': 0.0001,
    'patiente': 20,
    'monitoring_metric': 'val_acc',
    'normalization': '[0,1]',
}

manu_wave_normalize_1_1_20_window_bs_32_lr_5_e4_100_epochs = {
    'experiment_name': 'manu_wave_normalize_1_1_20_window_bs_32_lr_5_e4_100_epochs',
    'num_signals': 20,
    'arq': cnn_1d_manu(num_signals=20, num_features=63),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['val'],
    'l_users_val': d_users_split['val'],
    'l_users_test': d_users_split['test'],
    'batch_size': 32,
    'epochs': 1_000,
    'steps_per_epoch': 2_000,
    'lr': 0.0001,
    'patiente': 30,
    'monitoring_metric': 'val_f1_weig',
    'normalization': '[0,1]',
    'window_aug': True,
    'window_size': 20,
    'overlap': 5,
}



mia_cnn_normalize_1_1_bs_32_lr_5_e4_100_epochs = {
    'experiment_name': 'mia_cnn_normalize_1_1_bs_32_lr_5_e4_100_epochs',
    'num_signals': 20,
    'arq': cnn_1d_signal_classifier(num_signals=20, num_features=63),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['train'],
    'l_users_val': d_users_split['train'],
    'batch_size': 32,
    'epochs': 1_000,
    'steps_per_epoch': 2_000,
    'lr': 0.0001,
    'patiente': 30,
    'monitoring_metric': 'val_f1_weig',
    'normalization': '[-1,1]',
    'window_aug': False,
    'window_size': 20,
    'overlap': 5,
}

tfg_3gru_normalize_1_1_bs_32_lr_5_e4_100_epochs = {
    'experiment_name': 'tfg_3gru_normalize_1_1_bs_32_lr_5_e4_100_epochs',
    'num_signals': 20,
    'arq': cnn_1d_signal_classifier(num_signals=20, num_features=63),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['val'],
    'l_users_val': d_users_split['val'],
    'l_users_test': d_users_split['val'],
    'batch_size': 32,
    'epochs': 500,
    'steps_per_epoch': 50,
    'lr': 0.0001,
    'patiente': 30,
    'monitoring_metric': 'val_acc',
    'normalization': '[-1,1]',
    'window_aug': False,
    'window_size': 20,
    'overlap': 5,
}

svm_normalize_1_1 = {
    'experiment_name': 'svm_normalize_1_1',
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'num_signals': 20,
    'l_users_train': d_users_split['val'],
    'l_users_test': d_users_split['val'],
    'normalization': None,
    'window_aug': False,
}





f1_manu_wave_normalize_0_1_PCA_20_bs_32_lr_5_e4_ = {
    'experiment_name': 'f1_manu_wave_normalize_0_1_PCA_20_bs_32_lr_5_e4_',
    'window_size': 40,
    'num_features': 20,
    'arq': cnn_1d_manu(num_signals=40, num_features=20),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['val'],
    'l_users_val': d_users_split['val'],
    'l_users_test': d_users_split['val'],
    'batch_size': 32,
    'epochs': 1_000,
    'steps_per_epoch': 2_000,
    'lr': 0.0001,
    'patiente': 30,
    'monitoring_metric': 'val_f1_weig',
    'normalization': '[0,1]',
    'window_aug': False,
    'overlap': 5,
    'pca': True,
}

get_info = {
    'experiment_name': 'get_info',
    'window_size': 40,
    'num_features': 63,
    'arq': cnn_2d_manu(),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['train'],
    'l_users_val': d_users_split['val'],
    'l_users_test': d_users_split['test'],
    'batch_size': 32,
    'epochs': 1_000,
    'steps_per_epoch': 2_000,
    'lr': 0.0001,
    'patiente': 30,
    'monitoring_metric': 'val_f1_weig',
    'normalization': '[0,1]',
    'window_aug': True,
    'overlap': 15,
    'pca': False,
}

f1_manu_img_normalize_0_1_PCA_20_bs_32_lr_5_e4_ = {
    'experiment_name': 'f1_manu_img_normalize_0_1_PCA_20_bs_32_lr_5_e4_',
    'window_size': 40,
    'num_features': 20,
    'arq': cnn_1d_manu(num_signals=40, num_features=20),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['val'],
    'l_users_val': d_users_split['val'],
    'l_users_test': d_users_split['val'],
    'batch_size': 32,
    'epochs': 10,
    'steps_per_epoch': 1000,
    'lr': 0.0001,
    'patiente': 30,
    'monitoring_metric': 'val_f1_weig',
    'normalization': '[0,1]',
    'window_aug': False,
    'overlap': 5,
    'pca': True,
}


f1_mia_wave_normalize_0_1_PCA_20_bs_32_lr_5_e4_ = {
    'experiment_name': 'f1_mia_wave_normalize_0_1_PCA_20_bs_32_lr_5_e4_',
    'window_size': 40,
    'num_features': 20,
    'arq': cnn_1d_signal_classifier(num_signals=40, num_features=20),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['val'],
    'l_users_val': d_users_split['val'],
    'l_users_test': d_users_split['val'],
    'batch_size': 32,
    'epochs': 100,
    'steps_per_epoch': 2_000,
    'lr': 0.0001,
    'patiente': 30,
    'monitoring_metric': 'val_f1_weig',
    'normalization': '[0,1]',
    'window_aug': False,
    'overlap': 5,
    'pca': True,
}

f1_mia_lstm_normalize_0_1_PCA_20_bs_32_lr_5_e4_ = {
    'experiment_name': 'f1_mia_lstm_normalize_0_1_PCA_20_bs_32_lr_5_e4_',
    'window_size': 40,
    'num_features': 20,
    'arq': cnn_1d_signal_classifier(num_signals=40, num_features=20),
    'csv_pattern': 'zhang-wamsley-2019/data/CSV/*.csv',
    'l_users_train': d_users_split['train'],
    'l_users_val': d_users_split['val'],
    'l_users_test': d_users_split['val'],
    'batch_size': 32,
    'epochs': 100,
    'steps_per_epoch': 1000,
    'lr': 0.0001,
    'patiente': 30,
    'monitoring_metric': 'val_f1_weig',
    'normalization': '[0,1]',
    'window_aug': True,
    'overlap': 10,
    'pca': True,
}