from torch import nn

class SequentialNetwork(nn.Module):
    '''
    A custom sequential network class (as present in PyTorch) in which each 
    network based on user-defined specifications is instantiated as a 
    SequentialNetwork.
    
    This class also allows for appropriate save, train, and test modules for
    each model to be used as and when necessary.
    '''
   
    def __init__(self, network_list, names):
        '''
        Parameters
        ----------
        network_list : List
            Contains all the nn.Layer and nn.Activation modules to be added
            to the model.
        names : List
            Contains a identifying string associated with each nn.Layer and 
            nn.Activation module being added to the model.

        Returns
        -------
        None.

        '''
        super().__init__()
        for name, module in zip(names, network_list):
            self.add_module(name, module)
    
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
        pass
    
    def train(self, X_train, y_train):
        pass
    
    def test(self, X_test, Y):
        pass