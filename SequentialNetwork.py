import torch
from torch import nn
from collections import OrderedDict

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
        Saves the state_dict (type OrderedDict) to the given path as a .pt
        file. This file can later be loaded into another instantiated model.

        Parameters
        ----------
        PATH : String
            Directs where the torch.save() function saves the state_dict for 
            the given model instance. This string contains folder/filename 
            information, and new folders/files are created as necessary.

        Returns
        -------
        None.

        """
        torch.save(self, PATH)
        
    def load_state_dict(self, state_dict):
        print('Loading...')
        if not isinstance(state_dict, OrderedDict):
            raise TypeError("Expected state_dict to be dict-like, got {}."
                            .format(type(state_dict)))
        """
        Get all layer names.
        Get all layers!!!!
        Pass (un)initialised param to layer.
        """
        
        loading_dict = dict()
        
        loading_dict.update(state_dict)
        
        self.state_dict() == loading_dict
# =============================================================================
#         keys = list() 
#         layer = list()
#         for key in state_dict.keys():
#             keys.append(key)
#             params.append(state_dict.get(key))
#         
#         self.names = keys
#         self.network_list = params
# =============================================================================
        
    
    def train(self, X_train, y_train):
        pass
    
    def test(self, X_test, Y):
        pass