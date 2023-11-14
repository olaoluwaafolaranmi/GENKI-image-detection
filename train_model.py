from forward_backward_pass import forward_backward_pass
from getting_and_init_the_data import get_all_data_loaders
from genkimodel import GENKIModel
from lenet5_3d import LeNet5_3D
from params import DEVICE, LEARNING_RATE, N_EPOCHS, BATCH_SIZE, PATIENCE

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from copy import deepcopy


def train_model() -> Module:
    train_loader, _ = get_all_data_loaders(BATCH_SIZE)
    # model = GENKIModel()
    model = LeNet5_3D()

    # Move the model to the appropriate device
    model = model.to(DEVICE)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    for epoch in range(N_EPOCHS):
        # Train
        model, train_loss, train_acc = forward_backward_pass(
            model, optimizer, train_loader, DEVICE)

        print(f'Epoch: {epoch:02} | Epoch Train Loss: {train_loss:.3f} | Epoch Train Acc: {train_acc:.3f}')
        

    # Save the model
    torch.save(model.state_dict(), f'model/genki_model.pt')
    return model


if __name__ == '__main__':
    train_model()