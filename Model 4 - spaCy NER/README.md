# spaCy NER Model

## Setup Environment

- Install PyTorch CUDA.
- Install spaCy for GPU training models.
- Install scikit-learn.

## Training

### Manual

1. Install PyTorch and spaCy with CUDA.
2. Run [`data_pre_processing.py`](data_pre_processing.py) to generate training, validation and test data from the raw_data.
3. Edit [`config.cfg`](config.cfg) to set the correct GPU ID, and Hyperparameters.
4. Open a terminal in the folder containing `config.cfg`.
5. Train the model with `python -m spacy train config.cfg --output ./output --gpu-id 0`.

### Automated

1. Open [`automated_training.py`](./automated_training.py).
2. Adjust the hyperparameter tuning settings at the top.
3. Navigate to the `Model 4 - spaCy NER` folder in the Terminal: `cd "Model 4 - spaCy NER"`.
4. Run the file: `python automated_training.py`

## Models

You can see all these trained models in the [`trained_models`](./trained_models/) folder, named with the format `model-[total num of documents]-[train_valid_test split]`. This folder contains the training config with the hyperparameters used.

## Evaluation

The models are evaluated by their F1-Score, Precision and Recall when predicting the test same set. The following table shows the best performing models.

| Model                                                  | Training Time * | F1-Score | Precision | Recall  | Notes                                                                        |
| ------------------------------------------------------ | --------------- | -------- | --------- | ------- | ---------------------------------------------------------------------------- |
| [6807 60-20-20](./trained_models/model-6807-60_20_20/) | 70 min          | 0.71074  | 0.77246   | 0.65816 | This model and below, split train test data with sklearn (matching baseline) |

_\* Approximate time taken to train on a mobile RTX 3070Ti_

All the training results can be seen [here](./Training-Results.md).