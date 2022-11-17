import torch
from torch import nn
from torchsummary import summary
from io import StringIO


class SequentialNetwork(nn.Module):
    '''
    A custom sequential network class (as present in PyTorch) in which each 
    network based on user-defined specifications is instantiated as a 
    SequentialNetwork.
    
    This class also allows for appropriate save, train, and test modules for
    each model to be used as and when necessary.
    '''
   
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
        
        self.INPUT_SIZE = network_list[0].in_features
        
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
    
    def __str__(self):
        # TODO: Create a string represenation of the current model instance.
        return "This is a print"
    
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
                
    
    def train(self, X_train, y_train):
        pass
    
    def test(self, X_test, Y):
        pass
    
    def summary(self, input_size=None):
        if input_size is None:
            summary(self, (1, self.INPUT_SIZE))
        else:
            summary(self, input_size)
        
        