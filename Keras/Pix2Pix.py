import cv2
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, Concatenate, BatchNormalization, LeakyReLU, Dense
from keras.optimizers import Adam
from tqdm import tqdm
import gc
from keras.models import load_model
from keras.models import model_from_json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def dataIterator(x, y, batch_size):
    """
    From great jupyter notebook by Tim Sainburg:
    http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
    This contains some changes
    Args:
        data:
        batch_size:

    Returns:

    """

    while True:
        length = x.shape[0]  #
        idxs = np.arange(0, length)
        np.random.shuffle(idxs)
        for batch_idx in range(0, length, batch_size):
            if batch_idx + batch_size > length:
                break
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            batch_x, batch_y = x[cur_idxs], y[cur_idxs]
            yield batch_x, batch_y


def preprocess_data(train_path,
                    validation_path,
                    train_size=50000):
    print('')
    train_images = np.array(
        [cv2.imread(r'{}/{}'.format(train_path, x), 1) for x in os.listdir(train_path)[:train_size]])

    train_images = train_images / 127.5 - 1.

    train_abstract_images = train_images[:, :, :256, :]
    gc.collect()
    train_full_images = train_images[:, :, 256:, :]
    gc.collect()

    validation_images = np.array([cv2.imread(r'{}/{}'.format(validation_path, x), 1)
                                  for x in os.listdir(validation_path)])

    validation_images = validation_images / 127.5 - 1.

    validation_abstract_images = validation_images[:, :, :256, :]
    validation_full_images = validation_images[:, :, 256:, :]
    gc.collect()
    return train_abstract_images, train_full_images, validation_abstract_images, validation_full_images


def sample_images(real_abstract_images,
                  gan_full_images,
                  real_full_image,
                  path,
                  epoch):
    horizontal = []
    for rai, gfi, rfi in zip(real_abstract_images, gan_full_images, real_full_image):
        horizontal.append(np.concatenate(((rai + 1) * 127.5,
                                          (gfi + 1) * 127.5,
                                          (rfi + 1) * 127.5), axis=1))
    numpy_vertical_concat = np.concatenate(horizontal, axis=0)
    cv2.imwrite('{}/{}.jpg'.format(path, epoch), numpy_vertical_concat)
    # train_im = full_images[0].astype('uint8')


class Pix2Pix:

    def __init__(self, input_shape=(256, 256, 3),
                 discriminator_optimizer=Adam(lr=0.001),
                 generator_optimizer=Adam(lr=0.001)):
        self.input_shape = input_shape
        self.patch = int(input_shape[0] / 2 ** 4)
        self.disc_patch = (self.patch, self.patch, 1)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=discriminator_optimizer, loss='mse')

        # Gradients that will pass through the combined model will not update the discriminator
        self.discriminator.trainable = False
        for layer in self.discriminator.layers:
            layer.trainable = False

        self.generator = self.build_generator()

        # Create the combined GAN model
        abstract_img = Input(self.input_shape)
        generator_out = self.generator(abstract_img)
        discriminator_out = self.discriminator([abstract_img, generator_out])
        self.combined_model = Model(inputs=[abstract_img], outputs=[discriminator_out, generator_out])
        self.combined_model.compile(optimizer=generator_optimizer, loss=['mse', 'mae'], loss_weights=[0.01, 1])

    def build_discriminator(self):

        def discriminator_cd(input_tensor, filters, kernel_size, strides, batch_norm=False):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
            x = LeakyReLU(0.2)(x)
            x = BatchNormalization(momentum=0.8)(x) if batch_norm else x
            return x

        filters = 64
        abstract_img = Input(self.input_shape)
        full_img = Input(self.input_shape)

        x = Concatenate(axis=-1)([abstract_img, full_img])  # concatenate by channels

        x = discriminator_cd(x, filters, 4, 2, False)
        x = discriminator_cd(x, filters * 2, 4, 2, True)
        x = discriminator_cd(x, filters * 4, 4, 2, True)
        x = discriminator_cd(x, filters * 8, 4, 2, True)

        x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)

        return Model(inputs=[abstract_img, full_img], outputs=[x])

    def build_generator(self):

        def generator_encode_cd(input_tensor, filters, kernel_size, strides, batch_norm=False):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
            x = LeakyReLU(0.2)(x)
            x = BatchNormalization(momentum=0.8)(x) if batch_norm else x
            return x

        def generator_decode_cd(input_tensor, skip_tensor, filters, kernel_size, strides):
            x = UpSampling2D(size=2)(input_tensor)
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Concatenate()([x, skip_tensor])
            return x

        filters = 64
        abstract_img = Input(self.input_shape)

        e1 = generator_encode_cd(abstract_img, filters=filters, kernel_size=4, strides=2, batch_norm=False)
        e2 = generator_encode_cd(e1, filters=filters * 2, kernel_size=4, strides=2, batch_norm=True)
        e3 = generator_encode_cd(e2, filters=filters * 4, kernel_size=4, strides=2, batch_norm=True)
        e4 = generator_encode_cd(e3, filters=filters * 8, kernel_size=4, strides=2, batch_norm=True)
        e5 = generator_encode_cd(e4, filters=filters * 8, kernel_size=4, strides=2, batch_norm=True)
        e6 = generator_encode_cd(e5, filters=filters * 8, kernel_size=4, strides=2, batch_norm=True)
        e7 = generator_encode_cd(e6, filters=filters * 8, kernel_size=4, strides=2, batch_norm=True)

        d1 = generator_decode_cd(e7, e6, filters, 4, 1)
        d2 = generator_decode_cd(d1, e5, filters, 4, 1)
        d3 = generator_decode_cd(d2, e4, filters, 4, 1)
        d4 = generator_decode_cd(d3, e3, filters, 4, 1)
        d5 = generator_decode_cd(d4, e2, filters, 4, 1)
        d6 = generator_decode_cd(d5, e1, filters, 4, 1)

        d7 = UpSampling2D(size=2)(d6)
        d7 = Conv2D(filters=3, kernel_size=4, strides=1, padding='same', activation='tanh')(d7)

        return Model(inputs=[abstract_img], outputs=[d7])

    def train_on_batch(self, x,
                       y):
        if len(x) != len(y):
            raise ValueError('x and y must have the same batch size')

        prediction = self.generator.predict(x)
        # -----------------------
        # Train the discriminator
        # -----------------------

        # Train the discriminators to mark real data with 1's
        d_real_loss = self.discriminator.train_on_batch([x, y], np.ones((len(x),) + self.disc_patch))

        # Train the discriminators to mark generated data with 0's
        d_generated_loss = self.discriminator.train_on_batch([x, prediction], np.zeros((len(x),) + self.disc_patch))

        d_loss = 0.5 * (d_real_loss + d_generated_loss)

        # -----------------------
        # Train the generator
        # -----------------------

        g_loss = self.combined_model.train_on_batch(x, [np.ones((len(x),) + self.disc_patch), y])

        return d_loss, g_loss

    def fit(self, abstract_images,
            full_images,
            val_abstract_images,
            val_full_images,
            batch_size=8,
            epochs=50,
            verbose=True):

        if len(abstract_images) != len(full_images):
            raise ValueError('abstract_images and full_images must have the same size')

        data_iterator = dataIterator(abstract_images, full_images, batch_size)
        batches_per_epoch = len(abstract_images) // batch_size

        batches_per_epoch += 1 if batches_per_epoch < 1 else 0

        iteration = 0
        for epoch in range(1, epochs + 1):
            tqdm_ = tqdm(range(batches_per_epoch), disable=not verbose)
            for _ in tqdm_:
                x_current_batch, y_current_batch = next(data_iterator)
                d_loss, g_loss = self.train_on_batch(x_current_batch, y_current_batch)
                tqdm_.set_description("EPOCH: {}, D loss: {}, G loss: {}".format(epoch, d_loss, g_loss))
                if iteration % 50 == 0:
                    generated_images = self.generator.predict(val_abstract_images[:50])
                    sample_images(val_abstract_images[:50], generated_images[:50], val_full_images[:50],
                                  r'images', 'validation_{}'
                                  .format(iteration))

                    generated_images = self.generator.predict(abstract_images[:50])
                    sample_images(abstract_images[:50], generated_images[:50], full_images[:50],
                                  r'images', 'train_{}'
                                  .format(iteration))
                iteration += 1

if __name__ == '__main__':
    train_path = r'train_path'
    validation_path = r'val_path'

    train_abstract_images, train_full_images, validation_abstract_images, validation_full_images = \
        preprocess_data(train_path, validation_path, 300)

    p2p = Pix2Pix((256, 256, 3))
    p2p.fit(train_abstract_images, train_full_images, validation_abstract_images, validation_full_images, batch_size=8)
