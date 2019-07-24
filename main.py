from tools import *
import numpy as np
import pandas as pd

data = pd.read_csv("/Users/vladikturin/Desktop/pycharm_projects/recognition/train.csv")
Y = np.array(data.label).reshape([len(data.label), 1])
Y = np.array([np.eye(10)[el[0], :] for el in Y]).T

X = np.array(data.iloc[:, 1:]).T
X = X - np.mean(X) / np.std(X)
# print(X.shape)

# parameters = initialize_parameters((784, 500, 250, 100, 50, 10))
#
# AL, caches = L_model_forward(X, parameters, "dropout", 0.5)
#
# for key, value in caches.items():
#     print(f"{key}: shape of A_prev: {value[0].shape}, shape of Z: {value[1].shape}, shape of W: {value[2].shape}, shape of b: {value[3].shape}")


layer_dims = (784, 25, 10)
learning_rate = 0.001
num_iterations = 50

parameters1 = L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, "he")
parameters2 = L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, "normal")

AL1, caches1 = L_model_forward(X, parameters1)
AL2, caches2 = L_model_forward(X, parameters2)

accuracy1 = results(AL1, Y)
accuracy2 = results(AL2, Y)

print(accuracy1, accuracy2)
# X = pd.DataFrame(X)
# X[X > 0] = 1
# X[X <= 0] = 0
# print(np.array(X).mean())
