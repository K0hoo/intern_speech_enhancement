# Speech enhancement with deep learning
### dataset
* speech - TIMIT
* noise - NoiseX

Noise is divided seen and unseen noises. The seen noises are used to train the model.
Of course, the unseen noises are not used for training.

Datasets can be loaded by get_train_dataset() or get_test_dataset() in dataset_util.py.
Both of them are need transform parameter which is the dictionary type.
This dictionary requires 'mag_angle', 'logarithm', 'stft'.
All of value about those keys are boolean value.

get_train_dataset() returns (train_loader, validation_loader).

get_test_dataset() returns (seen_test_loader, unseen_test_loader).

### model
Define the new model in the model_*.py
There are two model file now. One of these is for LSTM and another is for TCN.

### mechanism (The format file is planned.)
The folders are classified by optimized stft or filter.

1. main.py gets the arguments and runs the main function in exec_<>.py.
2. exec_<>.py sets model, criterion, etc. Also, the hyperparameters should be set.
3. exec_<>.py runs train and test function in the execute.py
