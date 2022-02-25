import part1_nn_lib_copy as lib
import numpy as np

network = lib.LinearLayer(4,1)
x = np.array([[1,3,2,4], [1,2,4,2], [4,2,3,3]])
print(network.forward(x))