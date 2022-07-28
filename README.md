# Speech enhancement with deep learning
### dataset
* speech - TIMIT
* noise - NoiseX

Noise is divided seen and unseen noises. The seen noises are used to train the model.

### model
Define the new model in the model.py

### mechanism
1. main.py gets the arguments and runs the main function in exec_<>.py.
2. exec_<>.py sets model, criterion, etc. Also, the hyperparameters should be set.
3. exec_<>.py runs train and test function in the execute.py

Actual train and test function are in the execute.py. If you want to make new setting, create new exec_<>.py with exec_format.py.
