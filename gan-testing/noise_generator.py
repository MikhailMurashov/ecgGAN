import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    array = np.ndarray(shape=(10000, 1500))

    for i in tqdm(range(array.shape[0])):
        noise = np.random.rand(1, array.shape[1])
        np.append(array, noise)

    np.save('../data/noise', array)
