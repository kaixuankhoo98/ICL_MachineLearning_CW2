from part1_nn_lib import LinearLayer
import numpy as np

layer = LinearLayer(3,2)

# layer._W = np.array([4,2.5])
# layer._b = np.array([1.5,1.5])
# x = np.array([[2,3,4],[1,4,6],[5,6,7]])
x = np.array([[2,3,4],[1,4,6]])


y_hat = layer.forward(x)

print(y_hat)