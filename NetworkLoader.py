import torch
from torch import nn
from torch import optim

from NetworkSpecification import *
from SequentialNetwork import *

import re

# Pre-define training set
X = torch.Tensor([[0, 0, 0], [0, 0, 1], [0, 1, 1], 
                  [1, 0, 0], [1, 0, 1], [1, 1, 1]])
Y = torch.Tensor([2, 1, 0, 0, 1, 2])
README = 'For more information, consult README.md'


def read_and_train_models(filename):
    """
    TODO:
        Read models from file
        For each model, train on training set
        Save each trained model in the Trained folder
    """
    
    with open(filename, 'r') as file:
        for line, text in enumerate(file):
                
            # Check each line for reference to Untrained folder, load model
            if re.search('Untrained', text, flags=re.IGNORECASE): 
                loader = text.replace('\n', '')
                print('Identified model: {}'.format(text))
                model = torch.load(loader)
                
                # TODO: Train models
                model.train(X, Y)
                
                print('Model trained successfully: {}'
                      .format(model.successfully_trained))
                if model.successfully_trained:
                    saving_path = text.replace('Unt', 'T') #save at this path
                    model.save(saving_path)
                

def main():
    
    # Instantiate variables
    FILENAME = 'models.txt'

    print('Currently loading will be done from: {}'.format(FILENAME))
# =============================================================================
#     change = input('Would you like to specify another file? [y/n]\n>>> ')
#     if change == 'y':
#         FILENAME = input('Specify the name of the new file (with path if it is not in the current project folder).\n>>> ')
#         print('New file to load from: {}'.format(FILENAME))
#     
# =============================================================================
    print('The models will be trained on the following input: \nX: {}\ny: {}'
          .format(X, Y))
    print(README) #TODO: add to readme
    read_and_train_models(FILENAME)
    

if __name__ == "__main__":
    main()