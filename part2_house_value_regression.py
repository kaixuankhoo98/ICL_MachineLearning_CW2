import math
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F # includes key functions like loss function
import torch.optim as optim # updates model parameters using computed gradient
import pickle
import numpy as np
import pandas as pd
import read_data as rd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split, GridSearchCV # to ensure there's a held-out dataset...
import random
from itertools import product
import matplotlib.pyplot as plt

# # FROM THE COLAB e.g. You should set a random seed to ensure that your results are reproducible.
torch.manual_seed(0)
random.seed(0)

# # # Setting use of GPU
# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')
    
# print("Using GPU: {}".format(use_cuda))

""" 2 MAIN PARTS TO SORT! """

# Sources I've been working:
""" 
For pre-processing with SciPy
https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/
For example implementation of a one hidden layer classifier
https://www.youtube.com/watch?v=HXGmfzgZTr4
Converting pandas dataframe to numpy ndarray to use with Pytorch
https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor


 """

class Regressor(nn.Module):

    def __init__(self, x, y = None, nb_epoch = 100, batch_size = 32, neurons = [50], activations = ['relu']): 
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.
            - neurons is a list of integers, length is the number of layers
                and values are the number of neurons in corresponding layer
            - activations is a list of strings as activation functions,
                must match the length of neurons
              

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################


        assert(len(neurons) == len(activations))
        super().__init__() # call constructor of superclass

        """ Moved Scalars into the constructor"""
        self.x_scaler = preprocessing.RobustScaler() 
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
            # see https://michael-fuchs-python.netlify.app/2019/08/31/feature-scaling-with-scikit-learn/ for explanation.
        self.y_scaler = preprocessing.RobustScaler()
        
        # pre-process the data
        x, _ = self._preprocessor(x, (y if isinstance(y, pd.DataFrame) else None), training = True)
        """ SORT OUT: This is expecting a tensor of torch.Size([11558, 13]) rather than (11558, 9), need to work out how to change"""
        self.input_size = x.shape[1]
        self.output_size = 1 
        """"""
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
       
        # self.layers is a list of all the layers in the network
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, neurons[0]))
        if activations[0] == 'relu':
            self.layers.append(nn.ReLU())
        if activations[0] == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        if activations[0] == 'tanh':
            self.layers.append(nn.Tanh())


        # append to neurons list for every extra layer of neurons
        for i in range(1,len(neurons)):
            self.layers.append(nn.Linear(neurons[i-1],neurons[i]))
            # nn.init.xavier_uniform_(self.layers[-1]) # assigning random weights
            if activations[i] == 'relu':
                self.layers.append(nn.ReLU())
            if activations[i] == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            if activations[i] == 'tanh':
                self.layers.append(nn.Tanh())
        # last layer

        self.layers.append(nn.Linear(neurons[-1],self.output_size))

        # no activation for output layer because we're predicting an unbounded score.
        self.criterion = nn.MSELoss()
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, inputs): 

        """ 
        The `forward` method to defines the computation that takes place
        on the forward pass. A corresponding  `backward` method, which
        computes gradients, is automatically defined!
        e.g. implementation given in ICL's deep learning tutorial for one hidden layer classifier
        """
        # https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
        # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
        
        out = self.layers[0](inputs)
        for layer in self.layers[1:]:
            out = layer(out)
        return out
    

    def ohe_categorical(self, x):

        self.label_binarizer = preprocessing.OneHotEncoder(handle_unknown='ignore')

        """
        Sets all unknown categories to 0 we make each parameter we use for preprocessing a field of the Regressor
        object so as to apply the same preprocessing parameters to the test dataset as well.
        """
       
        ocean_proximity = np.array(x['ocean_proximity'])
        encoded_ocean_prox = self.label_binarizer.fit_transform(ocean_proximity.reshape(-1, 1)).toarray()

        # drop categorical column and fill with dummy/encoded column
        x = x.drop(['ocean_proximity'], axis = 1)

        for i, dummy in enumerate(np.unique(ocean_proximity)):
            x[dummy] = encoded_ocean_prox[:,i]

        """
        Our test data did not have "ISLAND" in it, so the one hot encoding returns 12 features rather than 13 as in the test data
        This causes a problem for the scaler transformation later, so we are making sure Island is included 
        """
        
        ocean_prox_variables = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

        for prox in ocean_prox_variables:
            if prox not in x:
                x[prox] = np.zeros(len(x))

        return x


    
    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # encode textual values using one-hot encoding
        x = self.ohe_categorical(x)
        # handle missing values.
        x = x.fillna(x.mean()) 
        # new preprocessing values needed if model is training
        training_columns_to_normalize = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
                                        'population', 'households', 'median_income']
        # Normalise X
        if training:
            self.x_scaler = self.x_scaler.fit(x.loc[:, training_columns_to_normalize])
            x = self.x_scaler.transform(x.loc[:, training_columns_to_normalize])

        else:
            x = self.x_scaler.transform(x.loc[:, training_columns_to_normalize])
        x_tensor = torch.from_numpy(np.array(x)).float()
        # Normalise y if given
        if y is not None:
            testing_column_to_normalize = ['median_house_value']
            if training:
                self.y_scalar = self.y_scaler.fit(y.loc[:, testing_column_to_normalize])
                y = self.y_scaler.transform(y.loc[:, testing_column_to_normalize])   
            else:
                y = self.y_scaler.transform(y.loc[:, testing_column_to_normalize])
            y_tensor = torch.from_numpy(np.array(y)).float()
            return(x_tensor, y_tensor)

        # Return preprocessed x and y
        return x_tensor, (y if isinstance(y, pd.DataFrame) else None)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, optimizer = None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
            - optimizer -- optimizer that updates moral parameters using computed gradient
        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        print("before pp")
        (X, Y) = self._preprocessor(x, y = y, training = True) 
        print("post")
        X = X.float()
        Y = Y.float()
        # prepare data for forward pass
        # use Pytorch utilities for data preparation
        print(1)
        dataset = torch.utils.data.TensorDataset(X, Y)

        average_loss_per_epoch = []

        # set model to training mode

        self.train()
        print(2)
        for epoch in range(self.nb_epoch):
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            total_loss_per_epoch = 0.0
            print(self.nb_epoch)
            for i, (input, labels) in enumerate(train_loader, 0):
                # forward pass
                if optimizer is not None:
                    optimizer.zero_grad()
                output = self(input)

                # calculate loss
                loss = (self.criterion(output, labels))
                loss.backward()
                
                # and optimize
                if optimizer is not None:
                    optimizer.step()
                
                total_loss_per_epoch += loss.item()
            average_training_loss = total_loss_per_epoch / len(train_loader)
            average_loss_per_epoch.append(average_training_loss)
            if epoch % 10 == 0:
                print("Epoch ", epoch, ", Average Training Loss ", average_training_loss)

        # code for plotting
        # plt.title("Training Loss")
        # plt.plot(range(len(average_loss_per_epoch)),average_loss_per_epoch)
        # plt.xlabel("Epoch number")
        # plt.ylabel("Average loss at each epoch")
        # plt.show()
        print("return")
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        print(1.1)
        X, _ = self._preprocessor(x, training = False) # Do not forget
        print(1.2)
        with torch.no_grad(): # for less memory consumption
            y_pred = self(X)            
        print(1.3)
        predictions = self.y_scaler.inverse_transform(y_pred)
        print(1.4)
        return predictions

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Underscore as want to ignore the parameter
        # Want to pre-process ground truths (y) in the same way pre-process x, but use the scalers that are stored (why trainging false) & just transform x & y
        x, _ = self._preprocessor(x, y = y, training = False) # Do not forget


        with torch.no_grad(): # for less memory consumption
            y_pred = self(x)            

        predictions = self.y_scaler.inverse_transform(y_pred)
        
        # calculate mse for predictions
        mse = mean_squared_error(y, predictions)
        # square root of mse
        rmse = math.sqrt(mse)
        print(rmse)
        return rmse 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)#to(device)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x,y): 

    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    looks for the best learning rate, batch size, and number of neurons

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """
    
    # setting params to test
    learning_rates = [0.1,0.001,0.0001,0.00001]
    batch_size = [16,32,64]
    neurons = [[120, 60],[5,5],[100, 50],[100]]
    activations = [['relu', 'sigmoid'], ['relu', 'sigmoid'], ['relu', 'sigmoid'],['relu']]

    min_error = 80000
    best_lr = 0
    best_bs = 0

    x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x, y, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.5)


    #Find best learning rate
    for lr in learning_rates:
        print("Testing learning rates:")
        print(lr)
        regressor = Regressor(x_train, y_train, nb_epoch = 100)#.to(device)
        # Create instance of optimizer
        optimizer = optim.SGD(regressor.parameters(), lr=lr, momentum=0.5) #TODO: not sure why we need a momentum

        regressor.fit(x_train, y_train, optimizer)
        save_regressor(regressor)

        
        # Error
        error = regressor.score(x_test, y_test)

        if error < min_error:
            min_error = error
            best_lr = lr
    print("Best learning rate: ", best_lr)
    print("Lowest error for lr: ", min_error)

    min_error = 80000


    #Use best learning rate and find best batch size
    for bs in batch_size:
        print("Testing batch size:")
        print(bs)
        regressor = Regressor(x_train, y_train, nb_epoch = 100, batch_size = bs)#.to(device)
        # Create instance of optimizer
        optimizer = optim.SGD(regressor.parameters(), lr=best_lr, momentum=0.5) #TODO: not sure why we need a momentum

        regressor.fit(x_train, y_train, optimizer)
        save_regressor(regressor)

        
        # Error
        error = regressor.score(x_test, y_test)

        if error < min_error:
            min_error = error
            best_bs = bs

    print("Best batch size: ", best_bs)
    print("with lowest error of: ", min_error)


    min_error = 80000


    #Use best learning rate and batch size and find best neuron config
    for i,j in zip(neurons, activations):
        print("Testing neurons size:")
        print(i)
        regressor = Regressor(x_train, y_train, nb_epoch = 100, batch_size = 64, neurons = i, activations =j)#.to(device)
        # Create instance of optimizer
        optimizer = optim.SGD(regressor.parameters(), lr=.1, momentum=0.5) 

        regressor.fit(x_train, y_train, optimizer)
        save_regressor(regressor)

        
        # Error
        error = regressor.score(x_test, y_test)

        if error < min_error:
            min_error = error
            best_n = i

    print("Best neurons: ", best_n)
    print("with lowest error of: ", min_error)



    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    ################## CODE TO UNDERSTAND the dataset ###################

    rd.first_and_last_five_rows(data)
    rd.summary_statistics(data)
    rd.dataset_datatypes(data)
    rd.missing_values(data)
    print(data.shape) 
    ################## PRE-PROVIDED CODE ###################

    # Spliting input and output
    X = data.loc[:, data.columns != output_label]
    Y = data.loc[:, [output_label]]
    # TRAINING
    # splitting out a held-out data set for validation and testing.
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    #x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.5)
    #TODO: think about whether we need x_val, y_val. Think we need it for hyperparameter tuning.
    #       we have training (70%), val (15%), and testing (15%) subsets for both x and y.

    print(x_train)

    # Prepocesses & learns appropriate transformation
    regressor = Regressor(x_train, y_train, nb_epoch = 100)#.to(device)
    # Create instance of optimizer
    optimizer = optim.SGD(regressor.parameters(), lr=0.01, momentum=0.5) #TODO: not sure why we need a momentum

    regressor.fit(x_train, y_train, optimizer)
    save_regressor(regressor)

        
    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error)) 


    regressor2 = Regressor(x_train, y_train, nb_epoch = 100, neurons = [100,50],batch_size = 64,activations = ['relu', 'sigmoid'])#.to(device)
    # Create instance of optimizer
    optimizer = optim.SGD(regressor2.parameters(), lr=0.1, momentum=0.5) #TODO: not sure why we need a momentum

    regressor2.fit(x_train, y_train, optimizer)
    save_regressor(regressor2)

        
    # Error
    error2 = regressor2.score(x_test, y_test)
    print("\nRegressor1 error: {}\n".format(error)) 
    print("\nRegressor2 error: {}\n".format(error2)) 


if __name__ == "__main__":
    example_main()
