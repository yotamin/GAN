import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from utils import dataIterator, sample_images
from tqdm import tqdm
import matplotlib.pyplot as plt

class AAE:

    def __init__(self, x_input_shape=(28, 28, 1),
                 aae_latent_dim=2,
                 discriminator_optimizer=Adam(0.001),
                 aae_optimizer=Adam(0.001),
                 aae_loss_weights=[0.999, 0.001]):

        self.x_input_shape = x_input_shape
        self.input_dim = x_input_shape[0] * x_input_shape[1] * x_input_shape[2]
        self.aae_latent_dim = aae_latent_dim

        self._discriminator = self.build_discriminator()
        self._discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
        self._discriminator.trainable = False
        for layer in self._discriminator.layers:
            layer.trainable = False

        self._encoder = self.build_encoder()
        self._decoder = self.build_decoder()

        input = Input(x_input_shape)
        encoded = self._encoder(input)
        decoded = self._decoder(encoded)

        encoded_on_discriminator = self._discriminator(encoded)

        self._aae = Model(inputs=input, outputs=[decoded, encoded_on_discriminator])
        self._aae.compile(optimizer=aae_optimizer, loss=['mse', 'binary_crossentropy'], loss_weights=aae_loss_weights)
        self._autoencoder = Model(inputs=input, outputs=decoded)

    def build_discriminator(self):
        input = Input((self.aae_latent_dim,))
        x = input
        for layer_dim in [self.aae_latent_dim * 6, self.aae_latent_dim * 3, self.aae_latent_dim]:
            x = Dense(units=layer_dim, activation=LeakyReLU(0.2))(x)
        output = Dense(units=1, activation='sigmoid')(x)
        return Model(inputs=input, outputs=output)

    def build_encoder(self):
        input = Input(self.x_input_shape)
        x = Flatten()(input)
        for layer_dim in [int(self.input_dim * .5)]:
            x = Dense(units=layer_dim, activation='relu')(x)
        # deteministic
        x = Dense(units=2, activation='linear')(x)
        return Model(inputs=input, outputs=x)

    def build_decoder(self):
        input = Input((self.aae_latent_dim,))
        x = input
        for layer_dim in [int(self.input_dim * .5)]:
            x = Dense(units=layer_dim, activation='relu')(x)
        x = Dense(units=self.input_dim, activation='tanh')(x)
        x = Reshape(self.x_input_shape)(x)
        return Model(inputs=input, outputs=x)

    def train_on_batch(self, x, latent_distribution_real_data):

        if len(x) != len(latent_distribution_real_data):
            raise ValueError('x and latent_distribution_real_data must have the same batch size')

        # Train the discriminator

        # Train the discriminator to mark real data with 1's
        d_real_loss = self._discriminator.train_on_batch(latent_distribution_real_data, np.ones(len(x)))

        # Train the discriminator to mark generated data with 0's
        d_generated_loss = self._discriminator.train_on_batch(self._encoder.predict(x), np.zeros(len(x)))

        d_loss = 0.5 * (d_real_loss + d_generated_loss)

        # Train the aae
        
        g_loss = self._aae.train_on_batch(x, [x, np.ones(len(x))])

        return d_loss, g_loss

    def fit(self, x, x_test, y_test, epochs=10, batch_size=32, verbose=True):
        data_iterator = dataIterator(x, batch_size)
        batches_per_epoch = len(x) // batch_size

        batches_per_epoch += 1 if batches_per_epoch < 1 else 0

        for epoch in range(1, epochs + 1):
            tqdm_ = tqdm(range(batches_per_epoch), disable=not verbose)
            for _ in tqdm_:
                x_current_batch = next(data_iterator)
                latent = np.random.normal(0, 1, (batch_size, self.aae_latent_dim))
                d_loss, g_loss = self.train_on_batch(x_current_batch, latent)
                tqdm_.set_description("EPOCH: {}, D loss: {}, G loss: {}".format(epoch, d_loss, g_loss))

            reconstructed_data = 0.5 * self._autoencoder.predict(x_current_batch[:25]) + 0.5
            sample_images(reconstructed_data, epoch, 'images')

            # Plot 2d latent
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            colors = ['blue', 'red', 'green', 'yellow', 'black', 'purple', 'orange', 'brown', 'grey', 'pink']
            for i, c in zip(range(10), colors):
                data = x_test[y_test == i]
                data = self._encoder.predict(data.reshape((len(data), 28, 28) + (1,)))
                ax.scatter(data[:, 0], data[:, 1], c=c, label=str(i))
            ax.grid(True)
            ax.legend(numpoints=10)
            fig.savefig('{}\\{}_latent.png'.format('images', epoch))
            plt.close()


if __name__ == '__main__':
    (X_train, _), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    aae = AAE()
    aae.fit(X_train, X_test, y_test, epochs=50)
