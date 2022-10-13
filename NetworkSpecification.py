class NetworkSpecification():
    '''
        This class stores Network Specification objects per desired/defined
        architecture. It stores the number of copies, the activations, and the 
        number of neurons per layer.
        The number of layers is implicitly computed by determining the length
        of the list of neurons.
        
        ?? This object also stores the init seed for each network ??
    '''
    def __init__ (self, N, activation, neurons):
        self.N = N
        self.activation = activation
        self.neurons = neurons
    
    def layers(self):
        """
        Computes the number of layers in the network

        Returns Number of Layers in network
        -------
        TYPE
            DESCRIPTION.

        """
        return len(self.neurons)
    
    def __str__(self):
        """
        Prints a pretty string outlining the particular network architecture.

        Returns ---
        -------
        None.

        """
        act = ""
        
        if type(self.activation) == str:
            act = "All layers in this network share the same activation function."
        else:
            act = "Each layer in the network has an individual activation function."
        return act