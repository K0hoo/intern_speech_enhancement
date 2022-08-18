# Speech enhancement with deep learning
### dataset
* speech - TIMIT
* noise - NoiseX

Noise is divided seen and unseen noises. The seen noises are used to train the model.
Of course, the unseen noises are not used for training.

There are two forms of data, the normal scale and log scale. The normal scaled data is not modified in the data level.
However, the log scaled data is cut in 2 second length. Also, the epsilon value(1e-8) is added not to be negative infinity.

### model
Define the new model in the model_*.py
There are two model file now. One of these is for LSTM and another is for TCN.

### mechanism (The format file is planned.)
1. main.py gets the arguments and runs the main function in exec_<>.py.
2. exec_<>.py sets model, criterion, etc. Also, the hyperparameters should be set.
3. exec_<>.py runs train and test function in the execute.py
