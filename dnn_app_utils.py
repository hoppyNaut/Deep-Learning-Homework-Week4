"""
主要步骤:
1.初始化网络参数
2.前向传播
(1).计算一层网络中线性变换部分
(2).计算激活函数的部分:sigmoid(1次)/relu(L-1次)
(3).结合线性变换和激活函数
3.计算误差
4.后向传播
(1).线性部分的反向传播公式
(2).激活函数部分的反向传播方式
(3).结合线性部分和激活函数部分的反向传播公式
"""
import numpy as np
import testCases
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward



def initialize_parameters(n_x, n_h, n_y):
    """
    初始化两层网络参数
    :param n_x:输入层节点数
    :param n_h:隐藏层节点数
    :param n_y:输出层节点数
    :return:
        parameters:包含参数的python字典
        W1 - 权重矩阵,维度为(n_h,n_x)
        b1 - 偏向量,维度为(n_h,1)
        W2 - 权重矩阵,维度为(n_y,n_h)
        b1 - 偏向量,维度为(n_y,1)
    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # 使用assert确保矩阵的格式正确
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def initialize_parameters_deep(layers_dims):
    """
    初始化多层网络中的参数
    :param layers_dims:包含每层网络的节点数的列表
    :return:
        parameters - 包含参数"w1","b1",....,"wL","bL"的字典
        Wi - 权重矩阵,维度为(layers_dims[i],layers_dims[i - 1])
        bi - 偏向量,维度为(layers_dims[i],1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for i in range(1, L):
        parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) / np.sqrt(layers_dims[i - 1])
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

        assert(parameters["W" + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
        assert(parameters["b" + str(i)].shape == (layers_dims[i], 1))

    return parameters


def linear_forward(A_prev, W, b):
    """
    前向传播中线性变换实现
    :param A_prev:来自上一层网络的输出矩阵作为该层的输入,维度为(上一层的节点数,示例数)
    :param W:当前网络层的权重矩阵,维度为(该层节点数,上一层的节点数)
    :param b:当前网络层的偏向量,维度为(该层节点数,1)
    :return:
        Z - 激活函数的输入
        cache - 包含了“A”,"W","b"的元组
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """

    :param activation:在此层所使用的激活函数名——“sigmoid”|”relu“
    :return:
        A - 激活函数的输出
        cache - 包含”linear_cache(包含了线性变换的输入)"和“activation_cache(包含了激活函数的输入)”的元组(A_prev, W, b, Z)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    实现多层网络的前向传播：L-1层Linear->Relu,最后一层Linear->Sigmoid
    :param X:输入的矩阵,维度(输入节点数量,示例数)
    :param parameters:initialize_parameters_deep()的返回值,包含了每一层的权重矩阵和偏向量
    :return:
        AL - 最后的激活值
        caches - 包含以下内容的缓存列表:
                    linear_relu_forward()的每个cache:L-1个,索引从0到L-2
                    linear_sigmoid_forward()的cache:1个,索引为L-1
    """
    caches = []
    A = X
    # // :表示a/b向下取整
    L = len(parameters) // 2
    # L-1层relu激活
    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(i)], parameters["b" + str(i)], "relu")
        caches.append(cache)

    # 最终层sigmoid激活
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    计算训练集的成本函数
    :param AL:与预测标签对应的结果向量,维度(1,示例数量)
    :param Y:标签向量,维度(1,示例数量)
    :return:
        cost - 成本
    """

    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    cost = np.squeeze(cost)

    assert(cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    线性部分的反向传播实现
    :param dZ:第l层线性部分的梯度
    :param cache:当前层前向传播的值的元组linear_cache(A_prev, W, b)
    :return:
        dW - 损失函数对该层W权重矩阵的梯度,维度与W相同
        db - 损失函数对该层偏向量b的梯度,维度与b相同
        dA_prev - 损失函数对于前一层激活函数输出A_prev的梯度,维度与A_prev相同
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T, dZ)

    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    激活函数和线性部分的反向传播实现
    :param dA:第l层激活函数输出的梯度
    :param cache:当前层前向传播的值的元组linear_cache + activation_cache(A_prev, W, b, Z)
    :return:
        dA_prev - 损失函数对于前一层激活函数输出A_prev的梯度,维度与A_prev相同
        dW - 损失函数对该层W权重矩阵的梯度,维度与W相同
        db - 损失函数对该层偏向量b的梯度,维度与b相同
    """
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    实现多层网络的后向传播：L-1层Linear->Relu,最后一层Linear->Sigmoid
    :param AL:最终的输出值,维度(1,示例数目)
    :param Y:标签向量,维度(1,示例数目)
    :param caches:包含以下内容的缓存列表:(linear_cache, activation_cache)
                linear_relu_forward()的每个cache:L-1个,索引从0到L-2
                linear_sigmoid_forward()的cache:1个,索引为L-1
    :returns:
        grads - 一个包含每层W, b, A梯度的字典
                grads["dA" + str(l)]、grads["dW" + str(l)]、grads["db" + str(l)]
    """

    grads = {}
    # 网络的层数
    L = len(caches)
    # 示例数
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_caches = caches[L - 1]
    # 计算最后一层的梯度
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)]  \
        = linear_activation_backward(dAL, current_caches, "sigmoid")

    for l in reversed(range(L - 1)):
        current_caches = caches[l]
        # print(str(l) + " ")
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 1)], current_caches, "relu")
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    更新每层网络的权重矩阵和偏向量
    :param parameters:initialize_parameters_deep()的返回值,包含了每一层的权重矩阵和偏向量
    :param grads:包含了损失函数对每一层权重矩阵和偏向量的梯度
    :param learning_rate:学习率
    :return:
        parameters:更新完后的参数字典
    """
    L = len(parameters) // 2
    for l in range(1,L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters



print("============测试initialize_parameters============")
parameters = initialize_parameters(3, 2, 1)
print(f'W1:{parameters["W1"]}')
print(f'b1:{parameters["b1"]}')
print(f'W2:{parameters["W2"]}')
print(f'b2:{parameters["b2"]}')

print("============测试initialize_parameters_deep============")
layers_dims = [4, 3, 2, 1]
parameters_deep = initialize_parameters_deep(layers_dims)
for i in range(1,len(layers_dims)):
    print(f'W{str(i)}:{parameters_deep["W" + str(i)]}')
    print(f'b{str(i)}:{parameters_deep["b" + str(i)]}')

print("============测试linear_forward============")
A, W, b = testCases.linear_forward_test_case()
Z, linecache = linear_forward(A, W, b)
print(f'Z = {Z}')

print("============测试linear_activation_forward============")
A_prev, W, b = testCases.linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print(f'sigmoid:A = {A}')
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print(f'relu:A = {A}')

print("============测试L_model_forward============")
X, parameters = testCases.L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print(f'AL = {AL}')
print(f'caches的长度为:{len(caches)}')

print("============测试compute_cost============")
Y, AL = testCases.compute_cost_test_case()
cost = compute_cost(AL, Y)
print(f'cost.shape:{cost.shape}')
print(f'cost:{cost}')

print("============测试linear_backforward============")
dZ, linear_cache = testCases.linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print(f'dA_prev:{dA_prev}')
print(f'dW:{dW}')
print(f'db:{db}')

print("============测试linear_activation_backforward============")
dA, linear_activation_cache = testCases.linear_activation_backward_test_case()
dA_prev, dW, db = linear_activation_backward(dA, linear_activation_cache, "sigmoid")
print("sigmoid:")
print(f'dA_prev:{dA_prev}')
print(f'dW:{dW}')
print(f'db:{db}')
dA_prev, dW, db = linear_activation_backward(dA, linear_activation_cache, "relu")
print("relu:")
print(f'dA_prev:{dA_prev}')
print(f'dW:{dW}')
print(f'db:{db}')

print("============测试L_model_backforward============")
AL, Y, caches = testCases.L_model_backward_test_case()
grads = L_model_backward(AL, Y, caches)
print(f'dW1:{grads["dW1"]}')
print(f'db1:{grads["db1"]}')
print(f'dA1:{grads["dA1"]}')

print("============测试update_parameters============")
parameters, grads = testCases.update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)
print(f'W1:{parameters["W1"]}')
print(f'b1:{parameters["b1"]}')
print(f'W2:{parameters["W2"]}')
print(f'b2:{parameters["b2"]}')