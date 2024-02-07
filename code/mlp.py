import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(
            self, 
            n_in:int, 
            n_out:int=10,
        ):
        """Create a MLP layer.

        Args:
            n_in (int): Number of dimensions for the input. The input is the positional encoding of the point.
            n_out (int): Number of dimensions for the output.
        """

        n_hidden = 128
        n_layers = 10

        super(MLP, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.layers = []
            
        # first layer
        self._add_layer(n_in, n_hidden)
        # hidden layers
        for _ in range(1,self.n_layers-1):    
            self._add_layer(n_hidden, n_hidden)
        # final layer has no activation
        self.layers.append(nn.Linear(in_features=n_hidden, out_features=n_out))

        self.mlp = nn.Sequential(*self.layers)

        self.init_weights()
            
    def _add_layer(self, n_in, n_out):
        """Add a layer to the MLP.
        
        Args:
            n_in (int): Number of input features.
        """

        self.layers.append(nn.Linear(in_features=n_in, out_features=n_out))
        self.layers.append(nn.ReLU())

    def init_weights(self):
        with torch.no_grad():
            self.layers[0].weight.uniform_(-1 / np.sqrt(self.n_in), 
                                                      1 / np.sqrt(self.n_in))
            for layer in range(1,self.n_layers):
                self.layers[2*layer].weight.uniform_(-1 / np.sqrt(self.n_hidden), 
                                                      1 / np.sqrt(self.n_hidden))

    def forward(self, in_node):
        """Forward pass of the MLP layer.

        Args:
            in_node (torch.Tensor): Input tensor of shape (:, n_in).
        Returns:
            out (torch.Tensor): Output tensor of shape (:, 2).
        """

        out = in_node.clone()

        # first and hidden layers
        for layer in range(self.n_layers-1):
            out = self.mlp[2*layer+1](self.mlp[2*layer](out))

        # final layer has no activation
        out = self.mlp[-1](out)

        return out
    
    def train(self, train_loader:DataLoader, num_epochs:int, device:str, verbose=True):
        """Train the model.

        Args:
            train_loader (Dataloader): Dataloader for the training set.
            num_epochs (int): Number of epochs.
            device (str): Device to use for training.
            verbose (bool): Print training progress.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
       
        losses = torch.zeros(num_epochs)

        for epoch in range(num_epochs):
            total_loss = 0

            for (inputs, targets) in train_loader:    
                inputs = inputs.to(device)
                targets = targets.to(device)

                # transform targets from [batch_size] to [batch_size, 10] with one hot encoding
                targets = nn.functional.one_hot(targets, num_classes=10).float()

                inputs = inputs.reshape(inputs.shape[0], -1)

                # Forward pass
                outputs = self.forward(inputs)

                # Compute loss
                loss = nn.MSELoss()(outputs, targets)
                total_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses[epoch] = total_loss

            if verbose:
                # Print training progress
                print('Epoch [{}/{}] Loss: {:f}.'
                .format(epoch+1, num_epochs, total_loss))

        return losses