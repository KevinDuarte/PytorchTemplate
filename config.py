from torch import cuda

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

n_epochs = 10
batch_size = 100

learning_rate = 1e-3
weight_decay = 1e-7

model_id = 1
save_dir = './SavedModels/Run%d/' % model_id