import os
import unittest
import numpy as np

import torch
from torch import nn
from torchsummary import summary

from NetworkGenerator import *
from NetworkLoader import *
from NetworkSpecification import *
from SequentialNetwork import *

class TestMR(unittest.TestCase):
    
    def test_network_spec(self):
        N = 2
        activation_str = 'relu'
        activation_list = [None, None, 'relu']
        neurons = [3, 4, 3]
        num_layers = 3
        
        ns_str = NetworkSpecification(N=N, activation=activation_str, 
                                      neurons=neurons)
        ns_list = NetworkSpecification(N=N, activation=activation_list, 
                                       neurons=neurons)
        
        # Check equality for activation as string value
        self.assertEqual(ns_str.N, N)
        self.assertEqual(ns_str.activation, activation_str)
        self.assertEqual(ns_str.neurons, neurons)
        self.assertEqual(ns_str.layers(), num_layers)
        
        # Check equality for activation as a list
        self.assertEqual(ns_list.activation, activation_list)
        self.assertEqual(ns_list.layers(), num_layers)

    def test_seq_net(self):
        network_list = [nn.Linear(3, 3), nn.Linear(3, 4), nn.Sigmoid(), 
                        nn.Linear(4, 1)]
        network_input = torch.Tensor([[1, 1, 1]])
        
        custom_model = SequentialNetwork(network_list=network_list)
        torch_model = nn.Sequential(*network_list)
        
        self.assertEqual(custom_model.network_list, network_list)
        self.assertEqual(custom_model.summary(), 
                         summary(torch_model, 
                                 (network_input.shape[0], 
                                  network_input.shape[1])))
        


if __name__ == "__main__":
    unittest.main()
