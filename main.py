from tools import *
import numpy as np
import pandas as pd

data = pd.read_csv("/Users/alextyurin/Desktop/pycharm_projects/recognition/train.csv")
Y = np.array(data.label).reshape([len(data.label), 1])
Y = np.array([np.eye(10)[el[0], :] for el in Y]).T

X = np.array(data.iloc[:, 1:]).T
X = X - np.mean(X) / np.std(X)

layer_dims = (784, 25, 10)
learning_rate = 0.001
num_iterations = 500

parameters = L_layer_model(X, Y, layer_dims, learning_rate, num_iterations)

AL, caches = L_model_forward(X, parameters)

accuracy = results(AL, Y)

print(accuracy)
# X = pd.DataFrame(X)
# X[X > 0] = 1
# X[X <= 0] = 0
# print(np.array(X).mean())
