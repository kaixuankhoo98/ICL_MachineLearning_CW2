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
from sklearn.model_selection import train_test_split # to ensure there's a held-out dataset...

# # FROM THE COLAB e.g. You should set a random seed to ensure that your results are reproducible.
torch.manual_seed(0)
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

    def __init__(self, x, y = None, nb_epoch = 1000, batch_size = 32, n_hidden = 1): 
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

        """
        Removed below section as shouldn't be pre-processing in constructor
        """
        #pre-process the data
        # x_train, _ = self._preprocessor(
        #     x, (y if isinstance(y, pd.DataFrame) else None),
        #     training = True
        # )
        """"""

        #TODO not entirely sure what the input_size should be...
        # print("adsa x shape: ", x.shape) # (11558, 9)
        #print("adsa x_train shape: ", x_train.shape) # torch.Size([11558, 13])


        """ SORT OUT: This is expecting a tensor of torch.Size([11558, 13]) rather than (11558, 9), need to work out how to change"""
        self.input_size = x.shape[1]+4
        self.output_size = 1 
        """"""


        # because we're predicting a single median value.
        #TODO get clear on what the batch and epoch size should be and why.
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.linear1 = nn.Linear(
            in_features=self.input_size, out_features=n_hidden, bias=True
        )
        self.linear2 = nn.Linear(
            in_features=n_hidden, out_features=self.output_size, bias=True
        )
        nn.init.xavier_uniform_(self.linear1.weight) # assigning random weights to connections
        nn.init.xavier_uniform_(self.linear2.weight)
        # no activation for output layer because we're predicting an unbounded score.
        self.criterion = nn.MSELoss()
        #TODO think about what loss function we're gonna use and why; just using MSELoss for now
                # set scalers

        """ Moved Scalars into the constructor"""
        self.x_scaler = preprocessing.RobustScaler() 
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
            # see https://michael-fuchs-python.netlify.app/2019/08/31/feature-scaling-with-scikit-learn/ for explanation.
        self.y_scaler = preprocessing.RobustScaler()
        """"""

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, inputs): # TODO: need to implement this cause we inherit from nn.Module
        #TODO: CHECK: i'm letting pytorch infer the batch size.
        # https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
        # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
        # feature_count = inputs.shape[0] * inputs.shape[1]
        out = self.linear1(inputs) #.view(-1, feature_count)
        out = torch.relu(out)
        out = self.linear2(out)
        return out
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

        """Our test data did not have "ISLAND" in it, so the one hot encoding returns 12 features rather than 13 as in the test data
        This causes a problem for the scaler transformation later, so we are making sure Island is included """
        ocean_prox_variables = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

        for prox in ocean_prox_variables:
            if prox not in x:
                x[prox] = np.zeros(len(x))
        """"""

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

        # print("trqw x shape:", x.shape)
        # print("trqw x:", x)
        # print(x.loc[:, ['ISLAND']])
        # encode textual values using one-hot encoding
        x = self.ohe_categorical(x)
        # print("trqw x shape:", x.shape)
        # handle missing values.
        x = x.fillna(x.mean()) #TODO: be able to explain why we fill the missing values with the mean.
        # print("trqw x shape:", x.shape)

        # new preprocessing values needed if model is training
        training_columns_to_normalize = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
                                        'population', 'households', 'median_income']

        # print("X: ", x)
        # print("y: ", y)
        
        """
        NEED TO SORT: not liking .loc, causes matrix multiplication error. Why? 
        """

        # Normalise X
        if training:
            # self.x_scaler = self.x_scaler.fit(x.loc[:, training_columns_to_normalize])
            # x = self.x_scaler.transform(x.loc[:, training_columns_to_normalize])
            self.x_scaler = self.x_scaler.fit(x)
            x = self.x_scaler.transform(x)

        else:
            # x = self.x_scaler.transform(x.loc[:, training_columns_to_normalize])
            x = self.x_scaler.transform(x)

        # convert X to tensor TO DO: COME BACK TO
        x_tensor = torch.from_numpy(np.array(x)).float()
        # print("x_tensor: ", x_tensor)

        # Normalise y if gievn
        if y is not None:
            testing_column_to_normalize = ['median_house_value']
            if training:
                # self.y_scalar = self.y_scaler.fit(y.loc[:,testing_column_to_normalize])
                # y = self.y_scaler.transform(y.loc[:,testing_column_to_normalize])])

                self.y_scalar = self.y_scaler.fit(y)
                y = self.y_scaler.transform(y)
                
            else:
                y = self.y_scaler.transform(y)


            # print("X post normalisation: ", x)
            # print("y post normalisation: ", y)
            # print("length of X: ", len(x), " & length of y: ", len(y))


            y_tensor = torch.from_numpy(np.array(y)).float()
            # print("y_tensor: ", y_tensor)

            return(x_tensor, y_tensor)

        # Return preprocessed x and y
        return x_tensor
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
        
        # print("In fit, before pre-processing, y: ", y)
        (X, Y) = self._preprocessor(x, y = y, training = True) # Do not forget
        # print("In fit, after pre-processing, X: ", X, " & Y: ", Y)
        # prepare data for forward pass
        # use Pytorch utilities for data preparation https://discuss.pytorch.org/t/what-do-tensordataset-and-dataloader-do/107017
        dataset = torch.utils.data.TensorDataset(X, Y)
        average_loss_per_epoch = []

        # set model to training mode: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        self.train()
        for epoch in range(self.nb_epoch):
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            total_loss_per_epoch = 0.0
            for i, (input, labels) in enumerate(train_loader, 0):
                # set our input and label tensors to use our device
                #input#.to(device)
                #labels#.to(device)
                # forward pass
                if optimizer is not None:
                    optimizer.zero_grad()
                output = self(input)

                # calculate loss
                loss = torch.sqrt(self.criterion(output, labels))
                loss.backward()
                
                # and optimize
                if optimizer is not None:
                    optimizer.step()
                
                total_loss_per_epoch += loss.item()
            average_training_loss = total_loss_per_epoch / len(train_loader)
            average_loss_per_epoch.append(average_training_loss)
            if epoch % 10 == 0:
                print("Epoch ", epoch, ", Average Training Loss ", average_training_loss)
        # TODO why is it going through the epochs so slowly?

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
        # BEING PASSED A TENSOR AS IT IS
        # X, _ = self._preprocessor(x, training = False) # Do not forget
        # x = torch.tensor(x).float()
        # turn on evaluation mode
        # self.eval()
        # predictions = []
        with torch.no_grad(): # for less memory consumption
            # for i, value in enumerate(x):
            #     outputs = self(value)
            #     predictions = np.append(predictions, outputs)
            y_pred = self(x)            

        predictions = self.y_scaler.inverse_transform(y_pred)
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
        # print("ahjsd score x.shape: ", x.shape, " & y.shape: ", y.shape)
        x, _ = self._preprocessor(x, y = y, training = False) # Do not forget
        # print("jhlk")
        # get prediction
        y_pred = self.predict(x)
        print(y_pred)
        
        # calculate mse for predictions
        mse = mean_squared_error(y, y_pred)
        # square root of mse
        rmse = math.sqrt(mse)
        return rmse # Replace this code with your own

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


if __name__ == "__main__":
    example_main()
