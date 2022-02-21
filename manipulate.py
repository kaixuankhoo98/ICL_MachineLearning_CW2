from part1_nn_lib import LinearLayer
import numpy as np

layer = LinearLayer(3,2)

# layer._W = np.array([4,2.5])
# layer._b = np.array([1.5,1.5])
x = np.array([[2,3,4],[1,4,6]])
# x = np.array([[2,3,2],[1,4,6],[1,9,2],[1,9,2]])

y_hat = layer.forward(x)

# print(y_hat)
grad_z = np.array([2,1,2])
layer.backward(grad_z)