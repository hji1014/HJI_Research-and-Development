from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, LeakyReLU
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import keras

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import mat73
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras import optimizers
seed = 0
np.random.seed(seed)

# 재매개변수화(reparameterization) 기법
# Q(z|X)에서 샘플링하는 대신에 eps = N(0,I)에서 샘플링을 실행
# 그때 z = z_mean + sqrt(var)*eps
def sampling(args):
    """등방성 단일 가우시안에서 샘플을 채취하는 재매개변수화 기법
    # Arguments:
        args (tensor): Q(z|X)의 분산의 로그값과 평균값
    # Returns:
        z (tensor): 샘플링된 은닉 벡터들
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # 기본설정으로, random_normal는 mean=0, std=1.0로 지정되있습니다.
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """2차원 은닉 벡터의 함수로서 라벨과 MNIST 숫자를 표시합니다
    # Arguments:
        models (tuple): 인코더와 디코더 모델
        data (tuple): 테스트 데이터와 라벨
        batch_size (int): 배치 사이즈
        model_name (string): 사용하려는 모델 이름
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
     # 은닉공간의 숫자 클래스의 2D 이미지를 표시합니다.
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # 30X30 2D형태의 숫자들을 표시합니다.
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # 은닉공간의 숫자 클래스의 2D 그림에 해당하는 선형 간격 좌표
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# # MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# image_size = x_train.shape[1]
# original_dim = image_size * image_size
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# dataset
path1 = 'C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/vae_input_Difficult1_0.2.mat'
path2 = 'C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/vae_input_Easy1_0.05.mat'

x_data = mat73.loadmat('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/vae_input_Difficult1_0.2.mat')
x_data = x_data['vae_input']
x_data = x_data[:, 0:41]

#x_train = x_data[:3100, :]
#x_test = x_data[3101:, :]

# 신경망 매개변수들
original_dim = x_data.shape[1]
input_shape = (x_data.shape[1], )
intermediate_dim = 32
batch_size = 64
latent_dim = 10
epochs = 300

# VAE model = encoder + decoder
# 인코더 모델 설계
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(inputs)
#x = Dense(intermediate_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(inputs)
x_16 = Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x)
z_mean = Dense(latent_dim, name='z_mean')(x_16)
z_log_var = Dense(latent_dim, name='z_log_var')(x_16)

# 재매개변수 기법을 이용해 샘플링을 입력으로 푸쉬합니다
# Tensorflow 백엔드에서는 "output_shape"이 필요하지 않습니다.
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# 인코더 모델을 인스턴스화(instantiate) 합니다.
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# 디코더 모델을 설계합니다.
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x_16 = Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(latent_inputs)
x = Dense(intermediate_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(x_16)
outputs = Dense(original_dim, activation='tanh', name='fake')(x)

# 디코더 모델을 인스턴스화 합니다.
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# VAE 모델을 인스턴스화 합니다.
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
vae.summary()

reconstruction_loss = mse(inputs, outputs)
#reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
Adam = optimizers.Adam(learning_rate=0.001)
vae.compile(optimizer=Adam)
vae.summary()
#early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
#vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))
vae.fit(x_data, epochs=epochs, batch_size=batch_size)
vae.save_weights('vae_mlp.h5')

# 1
#mean_model = keras.Model(inputs=vae.input, outputs=vae.get_layer('z_mean').output)
#mean_model_output = mean_model.predict(x_data)
#np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/mean_model_output.npy', mean_model_output)
# 2
#var_model = keras.Model(inputs=vae.input, outputs=vae.get_layer('z_log_var').output)
#var_model_output = var_model.predict(x_data)
#np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/var_model_output.npy', var_model_output)
# 3
# sampling_model = keras.Model(inputs=vae.input, outputs=vae.get_layer('z_sampling').output)
# sampling_model_output = sampling_model.predict(x_data)
# np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/sampling_model_output.npy', sampling_model_output)
# 4
fake_model = keras.Model(inputs=vae.input, outputs=vae.output)
fake_model_output = fake_model.predict(x_data)
np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/fake_model_output.npy', fake_model_output)

z_mean, z_log_var, z = encoder.predict(x_data, batch_size=batch_size)
np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/mean_model_output.npy', z_mean)
np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/var_model_output.npy', z_log_var)
np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/sample_model_output.npy', z)
outputs = vae.predict(x_data, batch_size=batch_size)
np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/fake_model2_output.npy', outputs)

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    help_ = "Load h5 model trained weights"
#    parser.add_argument("-w", "--weights", help=help_)
#    help_ = "Use mse loss instead of binary cross entropy (default)"
#    parser.add_argument("-m",
#                        "--mse",
#                        help=help_, action='store_true')
#    args = parser.parse_args()
#    models = (encoder, decoder)
#    #data = (x_test, y_test)
#
#    # VAE loss = mse_loss or xent_loss + kl_loss
#    if args.mse:
#        reconstruction_loss = mse(inputs, outputs)
#    else:
#        reconstruction_loss = binary_crossentropy(inputs,
#                                                  outputs)
#
#    reconstruction_loss *= original_dim
#    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#    kl_loss = K.sum(kl_loss, axis=-1)
#    kl_loss *= -0.5
#    vae_loss = K.mean(reconstruction_loss + kl_loss)
#    vae.add_loss(vae_loss)
#    vae.compile(optimizer='adam')
#    vae.summary()
#    #plot_model(vae,
#    #           to_file='vae_mlp.png',
#    #           show_shapes=True)
#
#    if args.weights:
#        vae.load_weights(args.weights)
#    else:
#        # 오토인코더를 학습합니다.
#        vae.fit(x_train,
#                epochs=epochs,
#                batch_size=batch_size,
#                validation_data=(x_test, None))
#        vae.save_weights('vae_mlp_mnist.h5')
#
#        # 1
#        mean_model = keras.Model(inputs=vae.input, outputs=vae.get_layer('z_mean').output)
#        mean_model_output = mean_model.predict(x_data)
#        np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/mean_model_output.npy', mean_model_output)
#        # 2
#        var_model = keras.Model(inputs=vae.input, outputs=vae.get_layer('z_log_var').output)
#        var_model_output = var_model.predict(x_data)
#        np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/var_model_output.npy', var_model_output)
#        # 3
#        # sampling_model = keras.Model(inputs=vae.input, outputs=vae.get_layer('z_sampling').output)
#        # sampling_model_output = sampling_model.predict(x_data)
#        # np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/sampling_model_output.npy', sampling_model_output)
#        # # 4
#        fake_model = keras.Model(inputs=vae.input, outputs=vae.output)
#        fake_model_output = fake_model.predict(x_data)
#        np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/fake_model_output.npy', fake_model_output)
#
#
#    # plot_results(models,
#    #              data,
#    #              batch_size=batch_size,
#    #              model_name="vae_mlp")
#
