import numpy as np


def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x / 255.0
    print('MNIST:', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os

    def parse_libsvm_line(line):
        parts = line.strip().split()
        label = int(parts[0])
        features = np.zeros(256)
        for part in parts[1:]:
            idx, val = part.split(':')
            features[int(idx) - 1] = float(val)
        return label, features

    train_path = os.path.join(data_path, 'usps')
    test_path = os.path.join(data_path, 'usps.t')

    data_train, labels_train = [], []
    with open(train_path, 'r') as f:
        for line in f:
            if line.strip():
                label, features = parse_libsvm_line(line)
                labels_train.append(label)
                data_train.append(features)

    data_test, labels_test = [], []
    with open(test_path, 'r') as f:
        for line in f:
            if line.strip():
                label, features = parse_libsvm_line(line)
                labels_test.append(label)
                data_test.append(features)

    x = np.concatenate([np.array(data_train), np.array(data_test)]).astype('float32')
    y = np.concatenate([np.array(labels_train), np.array(labels_test)])
    x = (x + 1) / 2.0
    x = x.reshape([-1, 16, 16, 1])
    print('USPS samples', x.shape)
    return x, y


def load_fashion_mnist():
    from keras.datasets import fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x / 255.0
    print('Fashion-MNIST:', x.shape)
    return x, y