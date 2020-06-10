import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from gan.model import create_gan
from gan.model import noise_length, ecg_shape

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable INFO massages from Tensorflow

norm_value = 200


def preprocess(data):
    result = np.ndarray((data.shape[0], ecg_shape[0]))
    x = 999
    for i in range(data.shape[0]):
        result[i] = data[i][x:x+ecg_shape[0]]

    result = result / norm_value

    return result


def load_data(path):
    print(f'Loading data from "{path}" ...')
    arr = np.load(path)

    data = np.ndarray(shape=(arr.shape[0], arr.shape[2]))
    for i in range(arr.shape[0]):
        data[i] = arr[i][0]

    return data


def show_results(generator):
    noise = np.random.rand(1, noise_length)
    ecgs = generator.predict(noise)

    denorm_ecgs = ecgs * norm_value

    plt.figure(figsize=(10, 4))
    plt.subplot(211)
    plt.plot(noise[0])
    plt.subplot(212)
    plt.plot(denorm_ecgs[0])
    plt.show()


def main():
    # data = preprocess(load_data('../data/signals.npy'))
    data = np.load('../data/expand_signals.npy')
    noise_arr = np.load('../data/noise.npy')

    data = data / norm_value

    generator, discriminator, gan = create_gan()

    batch_size = 64
    epochs = 200

    d_losses, d_accuracy, g_losses = [], [], []

    for epoch in tqdm(range(epochs)):
        idx = np.random.randint(0, data.shape[0], batch_size)
        true_ecgs = data[idx]

        idx = np.random.randint(0, noise_arr.shape[0], batch_size)
        noise = noise_arr[idx]

        fake_ecgs = generator.predict(noise)

        # prepare train data for discriminator
        X = np.concatenate([true_ecgs, fake_ecgs])
        y = np.zeros(2 * batch_size)
        y[:batch_size] = 1

        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y)
        discriminator.trainable = False

        # X = np.random.rand(batch_size, noise_length)
        y = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y)

        d_losses.append(d_loss[0])
        d_accuracy.append(d_loss[1])
        g_losses.append(g_loss)

    plt.plot(d_losses, 'r')
    plt.plot(g_losses, 'b')
    plt.show()

    # plt.plot(d_accuracy)
    # plt.show()

    show_results(generator)


if __name__ == '__main__':
    main()
