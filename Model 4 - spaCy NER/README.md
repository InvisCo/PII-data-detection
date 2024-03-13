# spaCy NER Model

## Training

1. Install PyTorch and spaCy with CUDA.
2. Run [`data_pre_processing.py`](data_pre_processing.py) to generate training, validation and test data from the raw_data.
3. Edit [`config.cfg`](config.cfg) to set the correct GPU ID, and Hyperparameters.
4. Open a terminal in the folder containing `config.cfg`.
5. Train the model with `python -m spacy train config.cfg --output ./output --gpu-id 0`.

## Models

You can see all these trained models in the [`trained_models`](./trained_models/) folder, named with the format `model-[total num of documents]-[train_valid_test split]`. This folder contains the training config with the hyperparameters used.

## Evaluation

This table shows the performance of the model when splitting it with different ratios and running it against different number of documents. 

| Train-Valid-Test Split                                                      | Number of Documents | Training Time * | F1-Score | Precision | Recall  |
| --------------------------------------------------------------------------- | ------------------- | --------------- | -------- | --------- | ------- |
| [60-20-20](./trained_models/model-6807-60_20_20/)                           | 6807                | 70 min          | 0.58843  | 0.62021   | 0.55975 |
| [60-20-20](./trained_models/model-6807-60_20_20/) (sklearn split test data) | 6807                | 70 min          | 0.71074  | 0.77246   | 0.65816 |
| [80(70-30)-20 A](./trained_models/model-1000-80(70_30)_20-A/)               | 1000                | 14 min          | 0.75482  | 0.82036   | 0.69898 |
| [80(70-30)-20 B](./trained_models/model-1000-80(70_30)_20-B/)               | 1000                | 11 min          | 0.61875  | 0.79839   | 0.5051  |
| [80(70-30)-20 C](./trained_models/model-1000-80(70_30)_20-C/)               | 1000                | 32 min          | 0.73889  | 0.81098   | 0.67857 |

_\* Approximate time taken to train on a mobile RTX 3070Ti_