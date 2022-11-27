import torch
from torch import autograd
from torch import nn
from torch import optim

from torchsummary import summary

import random
import numpy as np


# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class SequentialNetwork(nn.Module):
    '''
    A custom sequential network class (as present in PyTorch) in which each 
    network based on user-defined specifications is instantiated as a 
    SequentialNetwork.
    
    This class also allows for appropriate save, train, and test modules for
    each model to be used as and when necessary.
    '''
    
    loss_tracker = None
   
    def __init__(self, network_list=list()):
        '''
        Parameters
        ----------
        network_list : List
            Contains all the nn.Layer and nn.Activation modules to be added
            to the model.
            
        Returns
        -------
        None.

        '''
        super(SequentialNetwork, self).__init__()
        
        # Instantiate necessary variables
        self.INPUT_SIZE = network_list[0].in_features
        self.successfully_trained = False
        
        for name, module in enumerate(network_list):
            self.add_module(str(name), module)
    
    
    def forward(self, X):
        '''
        Carries out a forward pass through the model given the input X.

        Parameters
        ----------
        X : torch.Tensor
            Input to the model.

        Returns
        -------
        X : torch.Tensor
            A tensor obtained by passing the input through each consecutive
            layer in the model.

        '''
        for module in self.children():
            X = module(X)
        return X
    
    
    def save(self, PATH):
        """
        Saves the entire model to the given path as a .pt file. This file can 
        later be loaded using torch.load().

        Parameters
        ----------
        PATH : String
            Contains path to location on disk where the model is saved.

        Returns
        -------
        None.

        """
        torch.save(self, PATH)
                
    
    def summary(self, input_size=None):
        if input_size is None:
            summary(self, (1, self.INPUT_SIZE))
        else:
            summary(self, input_size)
    
    
    def train(self, X, Y, loss=nn.MSELoss(), optimiser=optim.SGD, 
              epochs=500, momentum=0.9, lr=0.05, decay=0, epsilon=1e-5): 
        
        self.loss_tracker = np.zeros(shape=(epochs+1))
        train_steps= X.size(0)

        opt = optimiser(self.parameters(), momentum=momentum, lr=lr, 
                        weight_decay=decay)
        
        epoch_print_interval = 0.1 * epochs
        
        # Training loop
        
        print('Training for {} epochs.'.format(epochs))
        
        for e in range(epochs + 1):
            for s in range(train_steps):
                
                datapoint = np.random.randint(X.size(0))
                x = autograd.Variable(X[datapoint], requires_grad=False)
                y = autograd.Variable(Y[datapoint], requires_grad=False)
                
                opt.zero_grad()
                
                y_pred = self.forward(x)
                
                loss_value = loss(y_pred, y)
                loss_value.backward()
                opt.step()
                
            
            self.loss_tracker[e] = loss_value.item()
            
            if e % epoch_print_interval == 0:
                print('Epoch: {}, Loss: {}'
                      .format(e, self.loss_tracker[e]))
        
        # TODO: Add more constraints, but this is a basic check for now
        if min(self.loss_tracker) <= epsilon:
            self.succesfully_trained = True