from part1_nn_lib import LinearLayer
import numpy as np
# from numpy import random_seed

# random_seed = 42

layer = LinearLayer(3,3)

# layer._W = np.array([4,2.5])
# layer._b = np.array([1.5,1.5])
x = np.array([[2,3,4],[1,4,6]])
# x = np.array([[2,3,2],[1,4,6],[1,9,2],[1,9,2]])

y_hat = layer.forward(x)

print("y_hat: ",y_hat)

# print(y_hat)
grad_z = np.array([2,1,2])
layer.backward(grad_z)

# x = 1 / (1 + np.exp(-x) )
print("x: ", x)

x[x<4] = -1

print("x: ", x)