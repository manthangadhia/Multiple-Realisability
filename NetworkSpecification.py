class NetworkSpecification():
    '''
        This class stores Network Specification objects per desired/defined
        architecture. It stores the number of copies, the activations, and the 
        number of neurons per layer.
        The number of layers is implicitly computed by determining the length
        of the list of neurons.
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
    