import torch

# check device
if torch.backends.mps.is_built():
    DEVICE = 'mps'
elif torch.has_cuda:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

if (DEVICE == 'cuda'):
    print(f'Device name: {torch.cuda.get_device_name(0)}', '\n\n')

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
N_EPOCHS = 40

IMG_SIZE = 32
N_CLASSES = 2

PATIENCE = 10