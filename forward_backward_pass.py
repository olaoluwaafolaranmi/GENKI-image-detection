import torch
import numpy as np
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_
from sklearn.metrics import accuracy_score


def forward_backward_pass(genki_model: Module, 
                        optim: Optimizer, 
                        dataloader: DataLoader, 
                        device: str, 
                        do_grad_clip: bool = False
                        ) -> tuple:

    iteration_loss = []

    image_targets = []
    image_predictions = []

    if optim is not None:
        # Indicate that we are in training mode
        genki_model.train()
    else:
        # Indicate that we are in evaluation mode
        genki_model.eval()

    for batch in dataloader:
        # Zero the gradient of the optimizer.
        if optim is not None:
            optim.zero_grad()

        # Get the batches.
        images, y_true = batch

        # Give them to the appropriate device & reshape the audio.
        images = images.float().to(device)
        y_true = y_true.to(device).float()

        # Get the prediction of the model
        y_hat = genki_model(images).flatten()

        # Calculate the loss of our model.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(input=y_hat, target=y_true)

        if optim is not None:
            # Do the backward pass
            loss.backward()

            # Gradient clipping (if applicable)
            if do_grad_clip:
                clip_grad_value_(genki_model.parameters(), 1)

            # Do an update of the weights (i.e. a step of the optimizer)
            optim.step()

        # Loss the loss of the batch
        iteration_loss.append(loss.item())

        image_targets.append(y_true.cpu())
        image_predictions.append(y_hat.detach().cpu())


    image_targets = torch.cat(image_targets)
    image_predictions = torch.cat(image_predictions)

    # Calculate accuracy metrics of the batch
    image_predictions_binary = (torch.sigmoid(image_predictions) > 0.5).float()
    accuracy = accuracy_score(y_true=image_targets.numpy(), y_pred=image_predictions_binary.numpy())

    return genki_model, np.mean(iteration_loss), accuracy