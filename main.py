#   tutorial: https://www.youtube.com/watch?v=jmQwYVeCUVI&list=PL-9x0_FO_lglas4qwPt2n-hgY2Wd3xKqs&index=1


import struct
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.special import expit


def load_data():
    with open('mnist database of handwritten digits\\train-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        # >II ... refers to big endian  (reason: dataset is stored in a format used by most non-intel processors
        # since we are using an intel processor we have flip the bytes in the header to big endian)
        train_labels = np.fromfile(labels, dtype=np.uint8)
    with open('mnist database of handwritten digits\\train-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        train_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)

    with open('mnist database of handwritten digits\\t10k-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)
    with open('mnist database of handwritten digits\\t10k-images.idx3-ubyte', 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
        test_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)
    return train_images, train_labels, test_images, test_labels


def visualize_data(img_array, label_array):
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(64):
        img = img_array[label_array == 9][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()


# train_x, train_y, test_x, test_y = load_data()

# visualize_data(train_x, train_y)

def enc_one_hot(y, num_labels=10):
    one_hot = np.zeros((num_labels, y.shape[0]))
    # the zeros function returns an array with the given dimensions filled with zeros
    for i, val in enumerate(y):
        one_hot[val, i] = 1.0
    return one_hot


# y = np.array([4, 5, 9, 0])
# z = enc_one_hot(y)
# print(y)
# print(z)

def sigmoid(z):
    # return (1 / (1 + np.exp(-z)))   # this is what the function looks like
    # there is no difference between expit(z) and the actual function
    return expit(z)


def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)


def visualize_sigmoid():
    # for very large negative values the functions tends towards 0
    # for very large positive values the functions tends towards 1
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fid, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()


# visualize_sigmoid()

def calc_cost(y_enc, outpt):
    t1 = -y_enc * np.log(outpt)
    t2 = (1 - y_enc) * np.log(1 - outpt)
    cost = np.sum(t1 - t2)
    return cost


def add_bias_unit(X, where):
    # where is just row of column
    if where == 'column':
        x_new = np.ones((X.shape[0], X.shape[1] + 1))
        x_new[:, 1:] = X
    elif where == 'row':
        x_new = np.ones((X.shape[0] + 1, X.shape[1]))
        x_new[1:, :] = X
    return x_new


def init_weights(n_features, n_hidden, n_output):
    # number of features 28x28(image size) 784

    # never init weights as 0, since 0 * something is always 0
    # if you initialize with 0 ... your NN is going to stall/ do nothing

    # this is moddeled for the shape of your neural network
    # in our case we have 1 input layer, 2 hidden layers and 1 output layer

    w1 = np.random.uniform(-1.0, 1.0, size=n_hidden * (n_features + 1))
    # np.random.uniform -> any value within the given interval is equally
    # likely to be drawn by uniform
    # returns n-dimensional array
    w1 = w1.reshape(n_hidden, n_features + 1)
    # reshape -> gives a new shape to an array without changing its data
    # 1. makes an array for each hidden layer (number of hidden layers)
    # 2. puts the n_features into the arrays
    # EXAMPLE:
    # a = np.array([1,2,3,4,5,6]).reshape(2,3)
    # OUTPUT:
    #   [[1 2 3]
    #    [4 5 6]]

    w2 = np.random.uniform(-1.0, 1.0, size=n_hidden * (n_hidden + 1))
    w2 = w2.reshape(n_hidden, n_hidden + 1)

    # connecting second hidden layer to output layer
    w3 = np.random.uniform(-1.0, 1.0, size=n_output * (n_hidden + 1))
    w3 = w3.reshape(n_output, n_hidden + 1)

    return w1, w2, w3


def feed_forward(x, w1, w2, w3):
    # add bias unit to the input
    # column within the row is just a byte of data
    # so we need to add a column of ones
    a1 = add_bias_unit(x, where='column')
    z2 = w1.dot(a1.T)
    a2 = sigmoid(z2)
    # since we have transposed we havet o add bias units as a row
    a2 = add_bias_unit(a2, where='row')
    z3 = w2.dot(a2)
    a3 = sigmoid(z3)
    a3 = add_bias_unit(a3, where='row')
    z4 = w3.dot(a3)
    a4 = sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, a4


def predict(x, w1, w2, w3):
    a1, z2, a2, z3, a3, z4, a4 = feed_forward(x, w1, w2, w3)
    y_pred = np.argmax(a4, axis=0)
    return y_pred


def calc_grad(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
    delta4 = a4 - y_enc
    z3 = add_bias_unit(z3, where='row')
    delta3 = w3.T.dot(delta4) * sigmoid_gradient(z3)
    delta3 = delta3[1:, :]
    z2 = add_bias_unit(z2, where='row')
    delta2 = w2.T.dot(delta3) * sigmoid_gradient(z2)
    delta2 = delta2[1:, :]

    grad1 = delta2.dot(a1)
    grad2 = delta3.dot(a2.T)
    grad3 = delta4.dot(a3.T)

    return grad1, grad2, grad3


# TODO: load save bias aswell?

def save_model(model, modelname):
    pickle.dump(model, open(modelname, 'wb'))


def load_model(modelname):
    weights = pickle.load(open(modelname, 'rb'))

    objects = []
    with (open(modelname, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    print(len(objects[0]))
    print(objects[0])

    # return

    return weights


def run_model(x, y, x_t, y_t, modelname=None):
    # x, y, x_t, y_t are the datasets ... _t being test datasets

    x_copy, y_copy = x.copy(), y.copy()
    y_enc = enc_one_hot(y)
    epochs = 30
    # epoch ... how many times do we train the model, iterations
    # (generally more -> better; too much could lead to overfitting)
    batch = 50
    # batch ... smaller number of data so we dont run through all the data, which would take a long time

    iterations_prev = 0

    # loading model, if possible
    try:

        iterations_prev, w1, w2, w3 = load_model(modelname)

        print("loading new model with " + str(iterations_prev) + " epochs")

    except FileNotFoundError:

        w1, w2, w3 = init_weights(784, 75, 10)

        print("no model was loaded, training new model from scratch")

    alpha = 0.001  # 10 ^ -3   original value: 0.001
    # learning rate, typical value; determines how large of a step we take in the parameter space
    eta = 0.001
    #
    dec = 0.00001
    # decreasing eta by a small number each epoch

    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)
    delta_w3_prev = np.zeros(w3.shape)

    total_cost = []

    pred_acc = np.zeros(epochs)

    for i in range(epochs):  # training the model here

        shuffle = np.random.permutation(y_copy.shape[0])
        x_copy, y_enc = x_copy[shuffle], y_enc[:, shuffle]
        eta /= (1 + dec * i)

        mini = np.array_split(range(y_copy.shape[0]), batch)

        for step in mini:
            # feed forward the model
            a1, z2, a2, z3, a3, z4, a4 = feed_forward(x_copy[step], w1, w2, w3)
            cost = calc_cost(y_enc[:, step], a4)

            total_cost.append(cost)
            # back propagate
            grad1, grad2, grad3 = calc_grad(a1, a2, a3, a4, z2, z3, z4, y_enc[:, step], w1, w2, w3)

            delta_w1, delta_w2, delta_w3 = eta * grad1, eta * grad2, eta * grad3

            w1 -= delta_w1 + alpha * delta_w1_prev
            w2 -= delta_w2 + alpha * delta_w2_prev
            w3 -= delta_w3 + alpha * delta_w3_prev

            delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2, delta_w3_prev

            # print(delta_w1.shape)

        y_pred = predict(x_t, w1, w2, w3)
        pred_acc[i] = 100 * np.sum(y_t == y_pred, axis=0) / x_t.shape[0]

        iterations = i + iterations_prev
        print('epoch #', iterations)

        # auto saving model
        # if i != 0 and i % 5 == 0:
        #     print("saving model epoch: " + str(iterations))
        #     model = [iterations, w1, w2, w3]
        #     save_model(model, modelname)
        if i == epochs - 1:
            model = [iterations, w1, w2, w3]
            save_model(model, modelname)

    return total_cost, pred_acc, y_pred


train_x, train_y, test_x, test_y = load_data()

cost, acc, y_pred = run_model(train_x, train_y, test_x, test_y, "model.pkl")

# visualizing results
x_a = [i for i in range(acc.shape[0])]
x_c = [i for i in range(len(cost))]
print('final prediction accuracy is: ', acc[9])  # length of acc arr must be epochs -1
plt.subplot(221)
plt.plot(x_c, cost)
plt.subplot(222)
plt.plot(x_a, acc)
plt.show()

# visualizing failed predictions
miscl_img = test_x[test_y != y_pred][:25]
correct_lab = test_y[test_y != y_pred][:25]
miscl_lab = y_pred[test_y != y_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i + 1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# visualize_sigmoid()

# additional user code

f = open('mnist database of handwritten digits\\t10k-images.idx3-ubyte', 'rb')
image_size = 28
num_images = 5
f.read(16)

buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

image = np.asarray(data[1]).squeeze()
plt.imshow(image)
# plt.show()  # showing single image from dataset
