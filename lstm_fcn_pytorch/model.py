import torch
import numpy as np
from sklearn.utils import shuffle

from lstm_fcn_pytorch.modules import LSTM_FCN

class Model():
    def __init__(self, x, y, units, dropout, filters, kernel_sizes):
    
        '''
        Implementation of time series classification model introduced in Karim, F., Majumdar, S., Darabi, H.
        and Chen, S., 2017. LSTM fully convolutional networks for time series classification. IEEE access,
        6, pp.1662-1669.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, length) where samples is the number of time series
            and length is the length of the time series.

        y: np.array.
            Class labels, array with shape (samples,) where samples is the number of time series.

        units: list of int.
            The length of the list corresponds to the number of recurrent blocks, the items in the
            list are the number of units of the LSTM layer in each block.

        dropout: float.
            Dropout rate to be applied after each recurrent block.

        filters: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the number of filters (or channels) of the convolutional layer in each block.

        kernel_sizes: list of int.
            The length of the list corresponds to the number of convolutional blocks, the items in the
            list are the kernel sizes of the convolutional layer in each block.
        '''
        
        # Check if GPU is available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Shuffle the data.
        x, y = shuffle(x, y)
        
        # Scale the data.
        self.x_min = np.nanmin(x, axis=0, keepdims=True)
        self.x_max = np.nanmax(x, axis=0, keepdims=True)
        x = (x - self.x_min) / (self.x_max - self.x_min)
        
        # Build the model.
        model = LSTM_FCN(
            input_length=x.shape[1],
            units=units,
            dropout=dropout,
            filters=filters,
            kernel_sizes=kernel_sizes,
            num_classes=len(np.unique(y))
        )
        
        # Save the data.
        self.x = torch.from_numpy(x).to(self.device).float()
        self.y = torch.from_numpy(y).to(self.device).long()

        # Save the model.
        self.model = model.to(self.device)
    
    def fit(self, learning_rate, batch_size, epochs, verbose=True):
        
        '''
        Train the model.
        
        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.
            
        batch_size: int.
            Batch size.
            
        epochs: int.
            Number of epochs.
            
        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''
        
        # Generate the training batches.
        dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x, self.y),
            batch_size=batch_size,
            shuffle=True
        )
        
        # Define the optimizer.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Define the loss function.
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Train the model.
        self.model.train(True)
        print(f'Training on {self.device}.')
        for epoch in range(epochs):
            for features, targets in dataset:
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                accuracy = (torch.argmax(torch.nn.functional.softmax(outputs, dim=-1), dim=-1) == targets).float().sum() / targets.shape[0]
            if verbose:
                print('epoch: {}, loss: {:,.6f}, accuracy: {:.6f}'.format(1 + epoch, loss, accuracy))
        self.model.train(False)
        
    def predict(self, x):
        
        '''
        Predict the class labels.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, length) where samples is the number of time series
            and length is the length of the time series.
            
        Returns:
        __________________________________
        y: np.array.
            Predicted labels, array with shape (samples,) where samples is the number of time series.
        '''
        
        # Scale the data.
        x = (x - self.x_min) / (self.x_max - self.x_min)
        
        # Get the predicted probabilities.
        p = torch.nn.functional.softmax(self.model(torch.from_numpy(x).to(self.device).float()), dim=-1)

        # Get the predicted labels.
        y = np.argmax(p.detach().cpu().numpy(), axis=-1)
        
        return y
