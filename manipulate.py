import part1_nn_lib_gus as lib
import numpy as np
# from numpy import random_seed

# random_seed = 42

# layer = LinearLayer(3,3)

# # layer._W = np.array([4,2.5])
# # layer._b = np.array([1.5,1.5])
# x = np.array([[2,3,4],[1,4,6]])
# # x = np.array([[2,3,2],[1,4,6],[1,9,2],[1,9,2]])

# y_hat = layer.forward(x)

# print("y_hat: ",y_hat)

# # print(y_hat)
# grad_z = np.array([2,1,2])
# layer.backward(grad_z)

# # x = 1 / (1 + np.exp(-x) )
# print("x: ", x)

# x[x<4] = -1

# print("x: ", x)

network = lib.MultiLayerNetwork(
input_dim=4, neurons=[16, 2], activations=["relu", "sigmoid"]
)

inputs = np.array([2,3,2,3])
# `inputs` shape: (batch_size, 4)
# `outputs` shape: (batch_size, 2)

layer1 = lib.LinearLayer(4,16)
print("Inputs: ", inputs)
output = layer1.forward(inputs)
print("Output: ", layer1.forward(output))

layer1_act = lib.Relu()
inputs = output
print("Inputs: ", inputs)
output = layer1_act.forward(inputs)
print("Output: ", layer1_act.forward(output))

# outputs = network(inputs)
# `grad_loss_wrt_outputs` shape: (batch_size, 2)
# `grad_loss_wrt_inputs` shape: (batch_size, 4)
# grad_loss_wrt_inputs = network.backward(grad_loss_wrt_outputs)
# network.update_params(learning_rate)