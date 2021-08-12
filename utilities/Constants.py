# Configuration for Hardware *******************************************************************************************
# if set 'USE_CPU' to True, CPU will be used for training deep learning classifiers.
# Otherwise, GPU will be used if available
USE_CPU = False
GPU_MEMO_ON_DEMAND = True

# Configuration for Datasets *******************************************************************************************
DATASETS = ['HRSC2016']

# Configuration for Models *********************************************************************************************
MODELS = ['cGAN', 'dcGAN', 'pix2pix', 'cycleGAN']

# Hyper-parameter configuration for Models *****************************************************************************
ITERATIONS = range(1, 3)
EPOCHS = 1
BATCH_SIZE = 1
BUFFER_SIZE = 400
SHAPE = [256, 256]
VERBOSE = 2