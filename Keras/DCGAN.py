import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from utils import dataIterator, sample_images
from tqdm import tqdm


class DCGAN:
    """
    Adjusted for mnist, can be altered to any other dataset with changes to D and G
    """

    def __init__(self, x_input_shape=(28, 28, 1),
                 g_latent_dim=100,
                 discriminator_optimizer=Adam(0.0001),
                 generator_optimizer=Adam(0.0001)):
        
        self.x_input_shape = x_input_shape
        self.g_latent_dim = g_latent_dim

        self._discriminator = self.build_discriminator()
        self._discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

        # Gradients that will pass through the combined model will not update the discriminator
        self._discriminator.trainable = False
        for layer in self._discriminator.layers:
            layer.trainable = False

        self._generator = self.build_generator()

        # Create the combined GAN model
        z = Input((self.g_latent_dim,))
        generator_out = self._generator(z)
        discriminator_out = self._discriminator(generator_out)
        self._combined_model = Model(inputs=z, outputs=discriminator_out)
        self._combined_model.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

    def build_discriminator(self):
        input = Input(self.x_input_shape)
        x = Conv2D(32, kernel_size=3, strides=2, padding="same")(input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(units=1, activation='sigmoid')(x)

        return Model(inputs=input, outputs=x)

    def build_generator(self):
        input = Input((self.g_latent_dim, ))
        x = Dense(units=(int(self.x_input_shape[0] / 4)) * int((self.x_input_shape[1] / 4)) * 20)(input)
        x = Reshape((7, 7, 20))(x)
        x = UpSampling2D(interpolation='nearest')(x)
        x = Conv2D(32, kernel_size=3, padding="same", activation='relu')(x)
        x = UpSampling2D(interpolation='nearest')(x)
        x = Conv2D(64, kernel_size=3, padding="same", activation='relu')(x)
        x = Conv2D(self.x_input_shape[2], kernel_size=3, padding="same", activation='tanh')(x)

        return Model(inputs=input, outputs=x)

    def train_on_batch(self, x,
                       z):

        if len(x) != len(z):
            raise ValueError('x and z must have the same batch size')

        # -----------------------
        # Train the discriminator
        # -----------------------

        # Train the discriminator to mark real data with 1's
        d_real_loss = self._discriminator.train_on_batch(x, np.ones(len(x)))

        # Train the discriminator to mark generated data with 0's
        d_generated_loss = self._discriminator.train_on_batch(self._generator.predict(z), np.zeros(len(z)))

        d_loss = 0.5 * (d_real_loss + d_generated_loss)

        # -----------------------
        # Train the generator
        # -----------------------

        g_loss = self._combined_model.train_on_batch(z, np.ones(len(z)))

        return d_loss, g_loss

    def fit(self, x,
            batch_size=32,
            epochs=1,
            verbose=True):

        data_iterator = dataIterator(x, batch_size)
        batches_per_epoch = len(x) // batch_size

        batches_per_epoch += 1 if batches_per_epoch < 1 else 0

        for epoch in range(1, epochs+1):
            tqdm_ = tqdm(range(batches_per_epoch), disable=not verbose)
            for _ in tqdm_:
                x_current_batch = next(data_iterator)
                z = np.random.normal(0, 1, (batch_size, self.g_latent_dim))
                d_loss, g_loss = self.train_on_batch(x_current_batch, z)
                tqdm_.set_description("EPOCH: {}, D loss: {}, G loss: {}".format(epoch, d_loss, g_loss))

            generated_images = 0.5 * self._generator.predict(np.random.normal(0, 1, (25, self.g_latent_dim))) + 0.5
            sample_images(generated_images, epoch, 'images')


if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, axis=3)

    gan = DCGAN()
    gan.fit(X_train, epochs=30)
