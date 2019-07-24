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


def initialize_parameters(layer_dims, initialization_parameter="he"):
    parameters = {}
    L = len(layer_dims)

    if initialization_parameter == "normal":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    elif initialization_parameter == "he":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1]) * 0.1
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_activation_forward(A_prev, W, b, activation, keep_prop):
    Z = np.dot(W, A_prev) + b
    D = np.random.rand(Z.shape[0], Z.shape[1])
    D = D >= keep_prop

    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "tanh":
        A = tanh(Z)

    A = A * D
    cache = (A_prev, Z, W, b)
    return A, cache


def L_model_forward(X, parameters, regular_type, keep_prop):
    caches = {}
    A = X
    L = len(parameters) // 2
    if regular_type == "none" or regular_type == "l2":
        for l in range(1, L):
            A_prev = A
            W = parameters["W" + str(l)]
            b = parameters["b" + str(l)]
            A, cache = linear_activation_forward(A_prev, W, b, "relu", 0)
            caches[str(l)] = cache

        W = parameters["W" + str(L)]
        b = parameters["b" + str(L)]
        AL, cache = linear_activation_forward(A, W, b, "sigmoid", 0)
        caches[str(L)] = cache

    elif regular_type == "dropout":
        for l in range(1, L):
            A_prev = A
            W = parameters["W" + str(l)]
            b = parameters["b" + str(l)]
            A, cache = linear_activation_forward(A_prev, W, b, "relu", keep_prop)
            caches[str(l)] = cache

        W = parameters["W" + str(L)]
        b = parameters["b" + str(L)]
        AL, cache = linear_activation_forward(A, W, b, "sigmoid", 0)
        caches[str(L)] = cache

    return AL, caches


def compute_cost(AL, Y, regular_type, parameters):
    m = Y.shape[1]
    cost = -(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)).sum() / m

    l2_cost = 0
    L = len(parameters) // 2

    for i in range(1, L):
        l2_cost += np.sum(parameters["W" + str(i)])

    cost += l2_cost

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


def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, initialization_parameter, regular_type="none", keep_prop=0):
    costs = []

    parameters = initialize_parameters(layer_dims, initialization_parameter)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters, regular_type, keep_prop)

        cost = compute_cost(AL, Y, regular_type, parameters)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        # if i & 100 == 0:
        #     costs.append(cost)
        #     print(cost)
        print(f"Epoch: {i}, error: {cost}")

    return parameters





