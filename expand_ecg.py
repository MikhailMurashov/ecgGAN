import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# arr = np.load('../data/fixed/fix_signals_0.npy')
# for a in arr:
#     plt.figure(figsize=(10, 4))
#     plt.plot(a)
#     plt.show()
# exit(-1)


if __name__ == '__main__':
    dim = 400
    step = 100
    data = np.load('../data/fixed/fix_signals_0.npy')
    print(data.shape)

    # Показать все отведения
    # fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 16))
    #
    # i = 0
    # for ax in axes.flatten():
    #     ax.plot(data[0][i])
    #     i += 1
    #
    # plt.show()
    # exit(-1)

    # data = np.ndarray(shape=(arr.shape[0], arr.shape[2]))
    # for i in range(arr.shape[0]):
    #     data[i] = arr[i][0]

    result = []
    for i in range(data.shape[0]):
        for j in range(22):
            x = 100 + j*step
            y = x + dim

            # plt.figure(figsize=(10, 4))
            # plt.plot(data[i][x:y])
            # plt.show()

            R_index = np.argmax(data[i][x:y])
            if 125 < R_index < 250:
                result.append(data[i][x:y])

    print(len(result))
    res_arr = np.array(result)
    print(res_arr.shape)

    np.save('../data/fix_signals_400.npy', res_arr)
