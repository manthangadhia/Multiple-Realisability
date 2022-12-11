import torch
from torch import nn

from NetworkSpecification import *
from SequentialNetwork import *

import random
import numpy as np
import os


# =============================================================================
# # Seed
# seed = 123
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# =============================================================================



UNTRAINED_FOLDER = 'Untrained'
TRAINED_FOLDER = 'Trained'
EXTENSION = ".pt" 
INPUT_SIZE = 3
OUTPUT_SIZE = 1
FILE_WITH_SAVED_MODELS = "models.txt"

def parse_specifications(user_input):
    """
    This function outputs a list of NetworkSpecification objects as per the 
    user's input.

    Parameters
    ----------
    user_input : List[<Num_copies, Activation_func, [Neurons]>]
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
    
    def check_activation(activation, neurons):
        """
        If List() activation is provided by user, this method checks that 
        the list is sufficiently defined, i.e. it has the same number of 
        elements as there are layers.

        Parameters
        ----------
        activation : List / String
            Contains activation functions desired by the user.
        neurons : List
            Contains information about the number and size of layers.

        Returns
        -------
        Boolean
            Whether the defined activation input is accurate.

        """
        if type(activation) is list:
            return len(activation) == len(neurons)
        return True
            
    
    # The number of elements (tuples) indicated the number of distinct
    # architectures.
    num_architectures = len(user_input)
    num_networks = 0
    
    network_specs = list()
    
    for arch in user_input:
        N, act, neur = arch
        
        # Check for accurate activation input
        if not check_activation(act, neur):
            print('error') #TODO
            
        network_specs.append(NetworkSpecification(N, act, neur))
        
        num_networks += N
    
    return num_architectures, num_networks, network_specs
    
     
def generate_networks(network_specifications, file_with_saved_models):
    """
    Generate nn.Sequential models based on the parsed input list of
    NetworkSpecification objects. Save each model in an individual text file 
    with adaptive naming scheme.

    Parameters
    ----------
    network_specifications : List
        A list containing NetworkSpecification objects.
    
    file_with_saved_models : String
        Name of the file where the PATH of the saved model is stored.

    Returns
    -------
    None.

    """
   
    ''' Define helper functions '''
    def instantiate_and_save_sequential_network(network_list, network_info, 
                                                file_with_saved_models):
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

        file_with_saved_models : String
            Name of the file where the PATH of the saved model is stored.
        
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
        write(PATH, file_with_saved_models)
        
        
        
    def get_activation(activation, index=None):
        
        # Default case where activation is a simple string
        func = activation
        
        # Special case where a list of activation functions was provided
        if type(activation) is list:
            func = activation[index]
       
        if func == 'sigmoid':
            return nn.Sigmoid()
        elif func == 'relu':
            return nn.ReLU()
        elif func == 'none' or func == None:
            pass
        else:
            raise NotImplementedError()
    
    
    ''' Unpack specification data and generate + save networks'''
    for spec_num, spec in enumerate(network_specifications):
     
    # Unpack specification
        N, activation, neurons = spec.N, spec.activation, spec.neurons
         
        
        # Create N networks with the given specification
        for n in range(N):
                    
            network_list = list()

            # Design choice was made to avoid using LazyLinear layers, and 
            # instead the output size of previous layers is manually inferred.
            for i, layer_size in enumerate(neurons):
                if i == 0:
                    network_list.append(nn.Linear(INPUT_SIZE, layer_size))                
                    network_list.append(get_activation(activation, i))
                else:
                    network_list.append(nn.Linear(neurons[i-1], layer_size))                
                    network_list.append(get_activation(activation, i))
            
            # Append final Linear layer corresponding with output_dims=1
            network_list.append(nn.Linear(neurons[-1], OUTPUT_SIZE))
             
            instantiate_and_save_sequential_network(network_list,
                                            (spec_num, n, spec.layers()), 
                                            file_with_saved_models)


def write_preamble(n_arch, n_net, filename):
    
    
    buffer = '-----\n'
    file_size = os.path.getsize(filename)
    if file_size > 0:
        """ If file is not empty, replace with an empty file"""
        open(filename, 'w').close()   
    
    
    with open(filename, 'a') as file:
        file.write('Number of networks: {}\n'.format(n_net))
        file.write('Number of architectures: {}\n'.format(n_arch))
        file.write('\n')
        file.write(buffer)
        
        


def write(PATH, filename):
    """
    Appends the PATH string to the given file.

    Parameters
    ----------
    PATH : String
        Path to the location where generated models are saved.
    filename : String
        Name of file where all the PATHs are being stored.

    Returns
    -------
    None.

    """

    
    path = PATH + "\n"
    with open(filename, "a") as file:
        file.write(path)


def main():
    # TODO: Ask user for network_specification input
    
    user_input = [(1, 'sigmoid', [4, 2, 3]), 
                  (1, [None, None, 'sigmoid'], [4, 2, 3])] 

    print('All model paths are being written to: {}'
          .format(FILE_WITH_SAVED_MODELS))
               
    n_arch, n_net, net_spec = parse_specifications(user_input)
    print('Identified {} architectures with {} total networks.'
          .format(n_arch, n_net))
    write_preamble(n_arch, n_net, FILE_WITH_SAVED_MODELS)
    generate_networks(net_spec, FILE_WITH_SAVED_MODELS)
    
if __name__ == "__main__":
    main()
