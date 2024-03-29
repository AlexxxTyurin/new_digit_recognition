from tools import *
import numpy as np
import pandas as pd
from keras.layers import Input, ZeroPadding2D, BatchNormalization, Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model

data = pd.read_csv("/Users/alextyurin/Desktop/pycharm_projects/recognition/train.csv")

m = data.shape[0]

Y = np.array(data.label).reshape([len(data.label), 1])
Y = np.array([np.eye(10)[el[0], :] for el in Y])

X = np.array(data.iloc[:, 1:])
X = X - np.mean(X) / np.std(X)
X = np.reshape(X, [m, 28, 28, 1])

Y_train = Y[:int(0.8 * m), :]
Y_dev = Y[int(0.8 * m):, :]

X_train = X[:int(0.8 * m), :]
X_dev = X[int(0.8 * m):, :]

print(X.shape)
print(Y.shape)


def gestures_model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)

    X = Conv2D(6, (5, 5), strides=(1, 1), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('tanh')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = Conv2D(16, (5, 5), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('tanh')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    X = Conv2D(120, (5, 5), strides=(1, 1), name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('tanh')(X)

    X = Flatten()(X)
    X = Dense(84, activation='sigmoid', name='fc1')(X)
    X = Dense(10, activation='sigmoid', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='gestures_model')

    return model


model = gestures_model(X_train.shape[1:])
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=2, batch_size=100)

predictions_train = model.evaluate(X_train, Y_train)

print("Loss in a training set = " + str(predictions_train[0]))
print("Accuracy in a training set= " + str(predictions_train[1]))

predictions_dev = model.evaluate(X_dev, Y_dev)

print("Loss in a training set = " + str(predictions_dev[0]))
print("Accuracy in a training set= " + str(predictions_dev[1]))