import part1_nn_lib_copy as lib
import numpy as np

x = np.array([[1,3,2,4], [1,2,4,2], [4,2,3,3]])
# network = lib.LinearLayer(4,1)
# print(network.forward(x))
# sigmoid = lib.SigmoidLayer()
# print(sigmoid(network.forward(x)))
# relu = lib.ReluLayer()
# print(relu(network.forward(x)))

MLnetwork = lib.MultiLayerNetwork(4, [2, 3, 1], ['relu', 'sigmoid', 'sigmoid'])
print(MLnetwork.forward(x))
print(MLnetwork.backward([0.2]))