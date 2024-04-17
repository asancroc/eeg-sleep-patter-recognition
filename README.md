# EEG Sleep Pattern Recognition

Repo created as my personal folder to run my TFM expreiments about EEG Sleep Pattern Recognition using Python.

## Installation

Before running any experiments, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

### Machine Learning Models
To run machine learning experiments with predefined models, navigate to the notebooks directory:

```bash
cd notebooks
```

Then open the Jupyter notebook:

```bash
jupyter notebook ml_training.ipynb
```

Follow the instructions within the notebook to execute the training and evaluation of machine learning models.


### Deep Learning Models
To run deep learning experiments, you need to define your experiment settings in experiments.py first.

After setting up your experiment, use the following command to start the training process:

```bash
python train.py --experiment "your_experiment_name"
```

Make sure to replace "your_experiment_name" with the name of your experiment as defined in experiments.py.

#### Evaluating Model Performance
After training your models, evaluate their performance by running:

```bash
python evaluate.py --experiment "your_experiment_name"
```

Again, replace "your_experiment_name" with the appropriate experiment name.