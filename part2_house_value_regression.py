import torch
import torch.nn as nn
import torch.nn.functional as F # includes key functions like loss function
import torch.optim as optim # updates model parameters using computed gradient
import pickle
import numpy as np
import pandas as pd
import read_data as rd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split # to ensure there's a held-out dataset...

# Sources I've been working:
""" 
For pre-processing with SciPy
https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/
For example implementation of a one hidden layer classifier
https://www.youtube.com/watch?v=HXGmfzgZTr4
Converting pandas dataframe to numpy ndarray to use with Pytorch
https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor


 """
# FROM THE COLAB e.g. You should set a random seed to ensure that your results are reproducible.
torch.manual_seed(0)
# Setting use of GPU
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
    
print("Using GPU: {}".format(use_cuda))

class Regressor(nn.Module):

    def __init__(self, x, y = None, nb_epoch = 1000, n_hidden = 1): 
        # n_hidden refers to the size of (i.e., number of neurons in) your hidden layers.
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        super().__init__() # call constructor of superclass
        # pre-process the data
        x_train, _ = self._preprocessor(
            x, (y if isinstance(y, pd.DataFrame) else None),
            training = True
        )
        #TODO: not entirely sure what the input_size should be...
        # Should it be the number of columns in the training dataset? (That's what it is now and was given in the spec I think)
        self.input_size = x_train.shape[1]
        # or should it be like x_train.shape[0] * x_train.shape[1] i.e., the feature count.
        # or should it be a tensor of like view(batch_size, -1)
        # edited input_size according to https://towardsdatascience.com/pytorch-layer-dimensions-what-sizes-should-they-be-and-why-4265a41e01fd
        self.output_size = 1 
        # because we're predicting a single median value.
        self.nb_epoch = nb_epoch
        self.linear1 = nn.Linear(
            in_features=self.input_size, out_features=n_hidden, bias=True
        )
        self.relu = F.Relu() # going to apply the Relu activation function to output of input layer 
        self.linear2 = nn.Linear(
            in_features=n_hidden, out_features=self.output_size, bias=True
        )
        # no activation for output layer because we're predicting an unbounded score.
        self.criterion = nn.MSELoss()
        #TODO: think about what loss function we're gonna use and why; just using MSELoss for now
        # https://pytorch.org/docs/stable/nn.html#loss-functions
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, inputs): # TODO: need to implement this cause we inherit from nn.Module
        #TODO: CHECK: i'm letting pytorch infer the batch size.
        # https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
        # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
        feature_count = inputs.shape[0] * inputs.shape[1]
        out = self.linear1(inputs.view(-1, feature_count))
        out = self.relu(out)
        out = self.linear2(out)
        return out
        #TODO: not sure if this is right.
    """ the `forward` method to defines the computation that takes place
     on the forward pass. A corresponding  `backward` method, which
      computes gradients, is automatically defined!
      e.g. implementation given in ICL's deep learning tutorial for one hidden layer classifier
       """


    def ohe_categorical(self, x):
        self.label_binarizer = preprocessing.OneHotEncoder(handle_unknown='ignore')
        #                   sets all unknown categories to 0 
        #                   we make each parameter we use for preprocessing a field of the Regressor
        #                   object so as to apply the same preprocessing parameters to the test dataset
        #                   as well.
        ocean_proximity = np.array(x['ocean_proximity'])
        encoded_ocean_prox = self.label_binarizer.fit_transform(ocean_proximity.reshape(-1, 1)).toarray()

        # drop categorical column and fill with dummy/encoded column
        x = x.drop(['ocean_proximity'], axis = 1)

        for i, dummy in enumerate(np.unique(ocean_proximity)):
            x[dummy] = encoded_ocean_prox[:,i]
        
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

        '''
       
        see https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding if it works.
        '''
        #TODO: what is the meaning of the 'training' flag?
        # encode textual values using one-hot encoding
        x = self.ohe_categorical(x)
        # handle missing values.
        x = x.fillna(x.mean()) #TODO: be able to explain why we fill the missing values with the mean.
        # normalising data
        self.min_max_scaler = preprocessing.MinMaxScaler()
        x_scale = self.min_max_scaler.fit_transform(x)
        # Return preprocessed x and y, return None for y if it was None
        return x_scale, (y if isinstance(y, pd.DataFrame) else None)
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
        
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        for epoch in range(self.nb_epoch):
            pass
            #TODO: fill in this pseudocode
            # prepare data for forward pass
            # calculate loss
            # backward propogation
            # and optimize
            """ e.g. optimizer.zero_grad()
                     loss.backward()
                     optimizer.step() """

        
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

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

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

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        return 0 # Replace this code with your own

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
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

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

    """ rd.first_and_last_five_rows(data)
    rd.summary_statistics(data)
    rd.dataset_datatypes(data)
    rd.missing_values(data)
    print(data.shape) """
    ################## PRE-PROVIDED CODE ###################

    # Spliting input and output
    X = data.loc[:, data.columns != output_label]
    Y = data.loc[:, [output_label]]
    # TRAINING
    # splitting out a held-out data set for validation and testing.
    x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(X, Y, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.5)
    #TODO: think about whether we need x_val, y_val. Think we need it for hyperparameter tuning.
    #       we have training (70%), val (15%), and testing (15%) subsets for both x and y.
    regressor = Regressor(x_train, y_train, nb_epoch = 10).to(device)
    # Create instance of optimizer
    optimizer = optim.SGD(regressor.parameters(), lr=0.01, momentum=0.5)

    """ regressor.fit(x_train, y_train, optimizer)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error)) """


if __name__ == "__main__":
    example_main()

