from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, RMSprop

optimizer = RMSprop(0.00001)
ecg_shape = (1500,)
noise_length = 1500


def create_d_cnn():
    input_ecg = Input(shape=ecg_shape)

    x = Conv1D(filters=10, kernel_size=120, strides=5)(input_ecg)
    x = MaxPooling1D(pool_size=46, strides=3)(x)

    x = Conv1D(filters=5, kernel_size=36, strides=3)(x)
    x = MaxPooling1D(pool_size=24, strides=3)(x)

    answer = Dense(1, activation='softmax')(x)

    discriminator = Model(inputs=input_ecg, outputs=answer)
    # discriminator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    return discriminator


def create_d_mpl():
    input_ecg = Input(shape=ecg_shape)

    x = Dense(1024)(input_ecg)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    answer = Dense(1, activation='sigmoid')(x)

    discriminator = Model(inputs=input_ecg, outputs=answer)
    discriminator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    return discriminator


def create_g():
    input_noise = Input(shape=(noise_length,))

    x = Embedding(1000, 128)(input_noise)

    x = Bidirectional(LSTM(100, return_sequences=True))(x)
    x = Bidirectional(LSTM(100))(x)

    x = Dropout(0.5)(x)

    output_ecg = Dense(ecg_shape[0], activation='tanh')(x)

    generator = Model(inputs=input_noise, outputs=output_ecg)
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    return generator


def create_gan():
    discriminator = create_d_mpl()
    # discriminator.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    # discriminator.summary()
    discriminator.trainable = False

    generator = create_g()
    # generator.summary()

    gan_input = Input(shape=(noise_length,))
    fake_ecg = generator(gan_input)
    gan_output = discriminator(fake_ecg)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    # gan.summary()

    return generator, discriminator, gan
