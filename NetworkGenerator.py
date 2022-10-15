import torch
from torch import autograd
from torch import nn
from torch import optim

import numpy as np

from NetworkSpecification import *


UNTRAINED_FOLDER = 'Untrained'


def parse_specifications(user_input):
    """
    This function outputs a list of NetworkSpecification objects as per the 
    user's input.

    Parameters
    ----------
    user_input : List
        User inputs a list of tuples containing desired network structures
        including the number of copies per architecture, and the structure
        of the given architecture in terms of activation functions and 
        number of neurons per layer.

    Returns
    -------
    num_architectures : int
        The total number of distinct architectures desired by user.
        
    num_networks : int
        The total number of distinct networks (to be) generated; acquired by
        summing N over all tuples.
        
    network_specs : List
        A list of NetworkSpecification objects based on the parsed input list
        so that a following NetworkGenerator function can make use of the
        information to make Sequential Linear-layered networks.

    """
    
    # The number of elements (tuples) indicated the number of distinct
    # architectures.
    num_architectures = len(user_input)
    num_networks = 0
    
    network_specs = list()
    
    for arch in user_input:
        N, act, neur = arch
        network_specs.append(NetworkSpecification(N, act, neur))
        
        num_networks += N
    
    return num_architectures, num_networks, network_specs
    
     
def generate_networks(network_specifications):
    """
    Generate nn.Sequential models based on the parsed input list of
    NetworkSpecification objects. Save each model in an individual rich text
    file with adaptive naming scheme.
    
    ?? Return path on disk to saved models ??

    Parameters
    ----------
    network_specifications : List
        A list containing NetworkSpecification objects.

    Returns
    -------
    None.

    """
   
    
    def sequential_and_save(network_list):
        """
        Given a list of nn.Layers, generate a sequential network and save it
        to disk.

        Parameters
        ----------
        network_list : List
            Contains torch.nn items corresponding to the layers and activation
            functions of the desired network.

        Returns
        -------
        None.
        Saves generated network model to disk.

        """
        
        model = nn.Sequential(*network_list)
        
        # torch.save
    
    def get_activation(activation, index):
        
        # Default case where activation is a simple string
        func = activation
        
        # Special case where a list of activation functions was provided
        if type(activation) is list:
            func = activation[index]
       
        ''' Use Try-Except to deal with possibilities of invalid input '''
        if func == 'sigmoid':
            return nn.Sigmoid()
        elif func == 'relu':
            return nn.ReLU()
        else:
            raise NotImplementedError()
    
    
    
    
    
    for spec in network_specifications:
     
    # Unpack specification
        N, activation, neurons = spec.N, spec.activation, spec.neurons
         
        network_list = list()
         
        for i, layer_size in enumerate(neurons):
             network_list.append(nn.LazyLinear(layer_size))
             network_list.append(get_activation(activation, i))
         
        # sequential_and_save(network_list)
     
    model = nn.Sequential(*network_list) 
    
    
    return model


input = [(1, 'sigmoid', [3, 4, 5, 2, 1])]
                
        