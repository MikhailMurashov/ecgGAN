from keras.backend import mean
from keras.layers import *
from keras.initializers import RandomNormal
from keras.models import Model

Z_SIZE = 100  # size of the random vector used to initialize G

def loss_d(y_true, y_pred):
    return mean(y_true * y_pred)


def create_d():
    weight_init = RandomNormal(mean=0., stddev=0.02)
    input_data = Input(shape=(5000,), name='input_data')

    x = Conv1D(filters=32, kernel_size=3, padding='same', name='conv1', kernel_initializer=weight_init)(input_data)
    x = MaxPool1D(pool_size=2)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.3)(x)

    x = Conv1D(filters=64, kernel_size=3, padding='same', name='conv_2', kernel_initializer=weight_init)(x)
    x = MaxPool1D(pool_size=1)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.3)(x)

    x = Conv1D(filters=128, kernel_size=3, padding='same', name='conv_3', kernel_initializer=weight_init)(x)
    x = MaxPool1D(pool_size=2)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.3)(x)

    x = Conv1D(filters=256, kernel_size=3, padding='same', name='conv_4', kernel_initializer=weight_init)(x)
    x = MaxPool1D(pool_size=1)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.3)(x)

    features = Flatten()(x)

    output_is_fake = Dense(units=1, activation='linear', name='output_is_fake')(features)

    return Model(inputs=[input_data], outputs=[output_is_fake], name='discriminator')


def create_g():
    DICT_LEN = 10
    EMBEDDING_LEN = Z_SIZE

    weight_init = RandomNormal(mean=0, stddev=0.02)
