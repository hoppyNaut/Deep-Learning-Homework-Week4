import numpy as np
import h5py
import matplotlib.pyplot as plt
import lr_utils
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 加载数据
train_x_orig, train_y, test_x_orig, test_y, classes = lr_utils.load_dataset()

# Example of a picture
# index = 7
# plt.imshow(train_x_orig[index])
# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

print(f'train_x_orig.shape:{train_x_orig.shape}')
print(f'tran_y.shape:{train_y.shape}')
print(f'test_x_orig.shape:{test_x_orig.shape}')
print(f'test_y.shape:{test_y.shape}')

# 图像转化为向量
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# train_x_flatten = train_x_orig.reshape(-1, train_x_orig.shape[0])
# test_x_flatten = test_x_orig.reshape(-1, test_x_orig.shape[0])


# 数据标准化
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

print(f'train_x.shape:{train_x.shape}')
print(f'test_x.shape:{test_x.shape}')


# 定义超参数
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        # Forward Propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, activation = "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation = "sigmoid")

        # compute cost
        cost = compute_cost(A2, Y)

        # init Backward Propagation
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))


        # Backward Propagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation = "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = "relu")

        # update parameters
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost
        if print_cost and i % 100 == 0:
            print(f'Cost after iteration {i}: {np.squeeze(cost)}')
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(X, Y, parameters):
    """
    预测L层网络的成功率
    """

    m = X.shape[1]
    L = len(parameters) // 2
    p = np.zeros((1, m))

    AL, caches = L_model_forward(X, parameters)

    for i in range(m):
        if AL[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print(f'准确度为：{np.sum((p == Y)) / m}')

    return p



def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost == True and i % 100 == 0:
            print(f'Cost after iteration {i}: {np.squeeze(cost)}')
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters



layers_dims = [12288, 20, 7, 5, 1]

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

# parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)

## START CODE HERE ##
my_image = "my_image.jpg" # change this to the name of your image file
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = my_image
# image = np.array(plt.imread(fname))
image = Image.open(fname)
# Image.fromarray:实现array到image的转换
# Image.resize(size, resample=0):修改图片尺寸
my_image = np.array(image.resize(size=(64, 64))).reshape((64 * 64 * 3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")