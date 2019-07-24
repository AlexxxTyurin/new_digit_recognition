import numpy as np
import pandas as pd


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    Z = pd.DataFrame(Z)
    Z[Z <= 0] = 0
    return np.array(Z)


def tanh(Z):
    return np.tanh(Z)


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z = np.dot(W, A_prev) + b
        A = sigmoid(Z)

    elif activation == "relu":
        Z = np.dot(W, A_prev) + b
        A = relu(Z)

    cache = (A_prev, Z, W, b)
    return A, cache


def L_model_forward(X, parameters):
    caches = {}
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches[str(l)] = cache

    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A, W, b, "sigmoid")
    caches[str(L)] = cache

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)).sum() / m

    return cost


def sigmoid_backward(dA, cache):
    A_prev, Z, W, b = cache
    dZ = dA * (sigmoid(Z) * (1 - sigmoid(Z)))

    return dZ


def relu_bacward(dA, cache):
    A_prev, Z, W, b = cache
    Z = pd.DataFrame(Z)
    Z[Z > 0] = 1
    Z[Z <= 0] = 0
    dZ = dA * np.array(Z)

    return dZ


def tanh_backward(dA, cache):
    A_prev, Z, W, b = cache
    dZ = dA * (1 - np.power(Z, 2))


def linear_activation_backward(dA, cache, activation):
    if activation == "relu":
        dZ = relu_bacward(dA, cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache)

    A_prev, Z, W, b = cache
    m = A_prev.shape[1]

    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T) / m
    db = dZ.sum(axis=1, keepdims=1) / m

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[str(len(caches))]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[str(l+1)]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return parameters


def results(AL, Y):
    al = pd.DataFrame(AL)
    al[al > 0.5] = 1
    al[al <= 0.5] = 0
    al = np.array(AL)
    print(np.array((al == Y), dtype=float).mean())



def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations):
    costs = []

    parameters = initialize_parameters(layer_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        # if i & 100 == 0:
        #     costs.append(cost)
        #     print(cost)
        print(f"Epoch: {i}, error: {cost}")

    return parameters





