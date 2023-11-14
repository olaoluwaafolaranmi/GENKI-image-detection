from getting_and_init_the_data import get_all_data_loaders
from params import BATCH_SIZE, DEVICE
from lenet5_3d import LeNet5_3D
from forward_backward_pass import forward_backward_pass

import torch


def test_lenet():
    # If dataset_name is 'mnist', then we will test the model on the MNIST dataset.
    # If dataset_name is 'cifar10', then we will test the model on the CIFAR-10 dataset.
    _, test_loader = get_all_data_loaders(BATCH_SIZE)
    model = LeNet5_3D()

    # Load the model
    model.load_state_dict(torch.load(f'model/lenet5.pt'))
    
    # Move the model to the appropriate device
    model = model.to(DEVICE)
    
    # Test
    model, test_loss, test_acc = forward_backward_pass(
        model, None, test_loader, DEVICE)
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')