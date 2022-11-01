import torch
from torch import autograd
from torch import nn
from torch import optim

import numpy as np

from NetworkSpecification import *
from SequentialNetwork import *


UNTRAINED_FOLDER = 'Untrained'
TRAINED_FOLDER = 'Trained'
EXTENSION = ".pt" 


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
   
    ''' Define helper functions '''
    def instantiate_and_save_sequential_network(network_list, network_info):
        """
        Given a list of nn.Layers, generate a sequential network and save it
        to disk.

        Parameters
        ----------
        network_list : List
            Contains torch.nn items corresponding to the layers and activation
            functions of the desired network.
        
        network_info : Tuple <num_spec, num_copy, num_layers>
            Contains information about the network being instantiated to be 
            used for generating a relevant file name.

        Returns
        -------
        None.
        Saves generated network model to disk.

        """

        # Instantiate network with list of layers and activation functions
        model = SequentialNetwork(network_list)
                
        # Generate name based on network (format: "archX_copyX_Xlayers")
        
        num_spec = "arch" + str(network_info[0])
        copy = "copy" + str(network_info[1])
        layers = str(network_info[2]) + "layers"
        filename = "_".join([num_spec, copy, layers]) + EXTENSION

        # Save network with given name in UNTRAINED folder
        PATH = "/".join([UNTRAINED_FOLDER, filename])
        
        
        model.save(PATH)
        
        
        
    def get_activation(activation, index=None):
        
        # Default case where activation is a simple string
        func = activation
        
        # Special case where a list of activation functions was provided
        if type(activation) is list:
            func = activation[index]
       
        # 
        if func == 'sigmoid':
            return nn.Sigmoid()
        elif func == 'relu':
            return nn.ReLU()
        else:
            raise NotImplementedError()
    
    
    ''' Unpack specification data and generate + save networks'''
    for spec_num, spec in enumerate(network_specifications):
     
    # Unpack specification
        N, activation, neurons = spec.N, spec.activation, spec.neurons
         
        network_list = list()
        
        # Create N networks with the given specification
        for n in range(N):
            
            # LazyLinear is used to allow for input_dims to be inferred during
            # the first forward pass through the network.
            for i, layer_size in enumerate(neurons):
                network_list.append(nn.LazyLinear(layer_size))                
                network_list.append(get_activation(activation, i))
            
            # Append final LazyLinear layer corresponding with output_dims=1
            network_list.append(nn.LazyLinear(1))
            
            print(network_list)
             
            instantiate_and_save_sequential_network(network_list,
                                            (spec_num, n, spec.layers()))


input = [(1, 'sigmoid', [3, 5, 2])]
                
n_arch, n_net, net_spec = parse_specifications(input)

generate_networks(net_spec)

# =============================================================================
# def main():
#     pass
#     
# 
# if __name__ == "__main__":
#     main()
# =============================================================================
