# spaCy NER Model

## Training

1. Install PyTorch and spaCy with CUDA.
2. Run [`data_pre_processing.py`](data_pre_processing.py) to generate training, validation and test data from the raw_data.
3. Edit [`config.cfg`](config.cfg) to set the correct GPU ID, and Hyperparameters.
4. Open a terminal in the folder containing `config.cfg`.
5. Train the model with `python -m spacy train config.cfg --output ./output`.