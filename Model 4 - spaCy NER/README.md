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

| Model (Number of Documents) (Train-Valid-Test Split)               | Training Time * | F1-Score | Precision | Recall  | Notes                                                                        |
| ------------------------------------------------------------------ | --------------- | -------- | --------- | ------- | ---------------------------------------------------------------------------- |
| [6807 60-20-20](./trained_models/model-6807-60_20_20/)             | 70 min          | 0.58843  | 0.62021   | 0.55975 | Split train/test data differently                                            |
| [6807 60-20-20](./trained_models/model-6807-60_20_20/)             | 70 min          | 0.71074  | 0.77246   | 0.65816 | This model and below, split train test data with sklearn (matching baseline) |
| [1000 80(70-30)-20 A](./trained_models/model-1000-80(70_30)_20-A/) | 14 min          | 0.75482  | 0.82036   | 0.69898 | Learn rate = 0.001                                                           |
| [1000 80(70-30)-20 B](./trained_models/model-1000-80(70_30)_20-B/) | 11 min          | 0.61875  | 0.79839   | 0.5051  | Learn rate = 0.01                                                            |
| [1000 80(70-30)-20 C](./trained_models/model-1000-80(70_30)_20-C/) | 32 min          | 0.73889  | 0.81098   | 0.67857 | Learn rate = 0.0001                                                          |
| [1000 80(70-30)-20 D](./trained_models/model-1000-80(70_30)_20-D/) | 13 min          | 0.75     | 0.82317   | 0.68878 | Learn rate = 0.0005                                                          |

_\* Approximate time taken to train on a mobile RTX 3070Ti_