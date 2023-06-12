import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import mat73
import scipy.io as sio

#(x_train, _), (x_test, _) = mnist.load_data()
x_data = sio.loadmat('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/AE_2DCNN/Easy1/Easy1_0.05/spike_img.mat')
#x_data = sio.loadmat('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/AE_2DCNN/Difficult1/Difficult1_0.2/spike_img.mat')
x_data = x_data['spike_img']
# x_train = x_data[:3201, :, :]
# x_test = x_data[3201:, :, :]
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 72, 72, 1))
# x_test = np.reshape(x_test, (len(x_test), 72, 72, 1))
x_data = x_data.astype('float32') / 255.
x_data = np.reshape(x_data, (len(x_data), 72, 72, 1))
input_img = keras.Input(shape=(72, 72, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
#x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = layers.MaxPooling2D((2, 2), padding='same')(x)
#x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Flatten()(x)
#x = layers.Dense(1024, activation='relu')(x)
encoded = layers.Dense(10, activation='relu', name='feature_extractor')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
#x = layers.Dense(128, activation='relu')(encoded)
#x = layers.Reshape((4, 4, 8))(x)
#x =layers.Dense(1024, activation='relu')(encoded)
x = layers.Dense(20736, activation='relu')(encoded)
x = layers.Reshape((36, 36, 16))(x)
#x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = layers.UpSampling2D((2, 2))(x)
#x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
#autoencoder = keras.Model(input_img, encoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(x_data, x_data,
                epochs=500,
                batch_size=128,
                shuffle=True,
                validation_data=(x_data, x_data))

feature_extractor_model = keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('feature_extractor').output)
feature_extractor_model_output = feature_extractor_model.predict(x_data)
decoded_imgs = autoencoder.predict(x_data)
decoded_imgs = np.reshape(decoded_imgs, (len(decoded_imgs), 72, 72))
np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/AE_2DCNN/Easy1/Easy1_0.05/extracted_output.npy', feature_extractor_model_output)
np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/AE_2DCNN/Easy1/Easy1_0.05/full_output.npy', decoded_imgs)
#np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/AE_2DCNN/Difficult1/Difficult1_0.2/extracted_output.npy', feature_extractor_model_output)
#np.save('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/AE_2DCNN/Difficult1/Difficult1_0.2/full_output.npy', decoded_imgs)


n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_data[i].reshape(72, 72))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#
#encoder = keras.Model(input_img, encoded)
#encoded_imgs = encoder.predict(x_test)
#
#n = 10
#plt.figure(figsize=(20, 8))
#for i in range(1, n + 1):
#    ax = plt.subplot(1, n, i)
#    plt.imshow(encoded_imgs[i].reshape((4, 4 * 8)).T)
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()
#
