import torch 
from torch.optim import Adam
from torch import cuda, no_grad
from torch.nn import BCELoss
from model import GENKIModel
from getting_and_init_the_data import get_all_data_loaders
import numpy as np
from copy import deepcopy
from utils import get_accuracy, plot





def main():


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    #instantiate model
    model = GENKIModel()
    model = model.to(device)
    optimizer = Adam(params=model.parameters(), lr=1e-3)
    loss_function = BCELoss()

    batch_size = 8

    #load data
    train_loader,val_loader, test_loader = get_all_data_loaders(batch_size=batch_size)

    print("train count: " ,len(train_loader.dataset))
    print("val count: " ,len(val_loader.dataset))
    print("test count: " ,len(test_loader.dataset))

    epochs = 50
    best_model = None
    lowest_val_loss = 1e10
    best_val_epoch = 0
    training_loss = []
    val_loss = []
    training_acc = []
    val_acc = []

    print("Start training....")
    for epoch in range(1 , epochs+1):
        epoch_loss_training = []
        epoch_loss_validation = []
        epoch_acc_training = []
        epoch_acc_validation = []

        model.train()

        for batch in train_loader:
            #zero the gradients of the optimizer
            optimizer.zero_grad()
            #get batches
            x, y = batch
            #give batches to appropriate device
            x = x.to(device)
            y = y.to(device)
            y = y.unsqueeze(1)

            # get pred from model
            y_hat = model(x)

            #calculate loss from model
            loss = loss_function(y_hat, y.float())

            #propagate the loss backwards
            loss.backward()

            #update weights
            optimizer.step()

            epoch_loss_training.append(loss.item())
            
            batch_acc = get_accuracy(y,y_hat)
            epoch_acc_training.append(batch_acc)

        model.eval()
        with no_grad():
            #validation process
            for batch in val_loader:
                #get batch
                x_val, y_val = batch
                y_val = y_val.unsqueeze(1)
                
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                y_hat = model(x_val)

                loss = loss_function(y_hat, y_val.float())

                epoch_loss_validation.append(loss.item())

                batch_acc = get_accuracy(y_val,y_hat)
                epoch_acc_validation.append(batch_acc)

        #mean losses
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()
        val_loss.append(epoch_loss_validation)
        training_loss.append(epoch_loss_training)

        #accuracies
        epoch_acc_training = sum(epoch_acc_training)/len(train_loader.dataset)
        epoch_acc_validation = sum(epoch_acc_validation)/len(val_loader.dataset)
        training_acc.append(epoch_acc_training)
        val_acc.append(epoch_acc_validation)
    

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_val_loss:
            lowest_val_loss = epoch_loss_validation
            best_model = deepcopy(model.state_dict())
            best_val_epoch = epoch

        
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f} | '
              f'Training acc: {epoch_acc_training:7.4f} | '
              f'Validation acc: {epoch_acc_validation:7.4f}')
        
            
    print(f'Best epoch {best_val_epoch} with loss {lowest_val_loss}', end='\n\n')

    # Process similar to validation.
    print('Starting testing', end=' | ')
    testing_loss = []
    testing_acc = []
    torch.save(best_model, 'model/best_model.pt')
    model.load_state_dict(best_model)
    model.eval()
    with no_grad():
        for batch in test_loader:
            # x_test, y_test = ? 
            x_test, y_test = batch

            y_test = y_test.unsqueeze(1)
            # Pass the data to the appropriate device.
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            # make the prediction
            # y_hat = ?
            y_hat = model(x_test)
            
            # Calculate the loss.
            # loss = ?
            loss = loss_function(y_hat, y_test.float())
            batch_acc = get_accuracy(y_val,y_hat)
            testing_acc.append(batch_acc)

            testing_loss.append(loss.item())

    testing_loss = np.array(testing_loss).mean()
    acc = sum(testing_acc)/len(test_loader.dataset)
    print(f'Testing loss: {testing_loss:7.4f}')
    print(f'Testing accuracy: {acc:7.4f}')
    
    plot(training_loss, val_loss)
    plot(training_acc,val_acc, type="accuracy")

                

if __name__ == '__main__':
    main()
    
    
