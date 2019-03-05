import numpy as np
import cv2
import os
from keras.applications import VGG19
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Dense, PReLU, Add, UpSampling2D
from tqdm import  tqdm


def plot_images(lr_images,
                sr_images,
                hr_images,
                path,
                epoch):
    horizontal = []
    for lr, sr, hr in zip(lr_images, sr_images, hr_images):
        horizontal.append(np.concatenate((cv2.resize(lr, (288, 288)), sr, hr), axis=1))
    numpy_vertical_concat = np.concatenate(horizontal, axis=0)
    # cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
    # cv2.waitKey()
    cv2.imwrite('{}\\{}.jpg'.format(path, epoch), numpy_vertical_concat)


def preprocess_images(images_path,
                      path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    if not os.path.exists('{}\\low'.format(path_to_save)):
        os.makedirs('{}\\low'.format(path_to_save))

    if not os.path.exists('{}\\high'.format(path_to_save)):
        os.makedirs('{}\\high'.format(path_to_save))

    for img_name in os.listdir(images_path):
        original = cv2.imread(os.path.join(images_path, img_name), 1)

        low = cv2.resize(original, (72, 72))
        high = cv2.resize(original, (288, 288))

        cv2.imwrite('{}\\low\\{}_low.jpg'.format(path_to_save, img_name.split('.')[0]), low)
        cv2.imwrite('{}\\high\\{}_high.jpg'.format(path_to_save, img_name.split('.')[0]), high)


def dataIterator(lr_data,
                 hr_data,
                 batch_size):
    """
    From great jupyter notebook by Tim Sainburg:
    http://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN
    This contains some changes
    Args:
        data:
        batch_size:

    Returns:

    """
    if len(lr_data) != len(hr_data):
        raise ValueError('lr_data and hr_data must have the same size')

    while True:
        length = lr_data.shape[0]
        idxs = np.arange(0, length)
        np.random.shuffle(idxs)
        for batch_idx in range(0, length, batch_size):
            if batch_idx + batch_size > length:
                break
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            batch = lr_data[cur_idxs], hr_data[cur_idxs]
            yield batch


class SRGAN:

    def __init__(self, discriminator_optimizer=Adam(0.001),
                 generator_optimizer=Adam(0.001),
                 generator_loss_weights=[1e-3, 1]):
        self.lr_shape = (72, 72, 3)
        self.hr_shape = (288, 288, 3)
        self.B_residual_blocks = 8

        # Discriminator
        self._discriminator = self.build_discriminator()
        self._discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
        self._discriminator.trainable = False
        for layer in self._discriminator.layers:
            layer.trainable = False
        # VGG
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=self.hr_shape)
        img_features = vgg(img)
        self._vgg = Model(img, img_features)
        self._vgg.trainable = False
        for layer in self._vgg.layers:
            layer.trainable = False

        self._generator = self.build_generator()

        lr_input = Input(self.lr_shape)
        # hr_input = Input(self.hr_shape)

        sr_img = self._generator(lr_input)
        g_on_discriminator = self._discriminator(sr_img)
        g_on_vgg_features = self._vgg(sr_img)

        self._combined_model = Model(inputs=[lr_input], outputs=[g_on_discriminator, g_on_vgg_features])
        self._combined_model.compile(optimizer=generator_optimizer,
                                     loss=['binary_crossentropy', 'mse'],
                                     loss_weights=generator_loss_weights)

    def build_discriminator(self):
        filters = 64
        input_img = Input(self.hr_shape)
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation=LeakyReLU(0.2))(input_img)
        x = Conv2D(filters=filters, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(0.2))(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(filters=filters * 2, kernel_size=3, strides=1, padding='same', activation=LeakyReLU(0.2))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(filters=filters * 2, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(0.2))(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same', activation=LeakyReLU(0.2))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(filters=filters * 4, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(0.2))(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Conv2D(filters=filters * 8, kernel_size=3, strides=1, padding='same', activation=LeakyReLU(0.2))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(filters=filters * 8, kernel_size=3, strides=2, padding='same', activation=LeakyReLU(0.2))(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Dense(units=1024, activation=LeakyReLU(0.2))(x)
        x = Dense(units=1, activation='sigmoid')(x)

        return Model(inputs=input_img, outputs=x)

    def build_generator(self):
        filters = 64

        def residual_block(input):
            x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input)
            x = BatchNormalization(momentum=0.8)(x)
            x = PReLU()(x)
            x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            return Add()([input, x])

        input_img = Input(self.lr_shape)
        x = Conv2D(filters=filters, kernel_size=9, strides=1, padding='same')(input_img)
        x = PReLU()(x)

        residual = residual_block(x)
        for _ in range(self.B_residual_blocks-1):
            residual = residual_block(residual)

        residual = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(residual)
        residual = BatchNormalization(momentum=0.8)(residual)

        x = Add()([x, residual])

        # Upsample
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = UpSampling2D(size=2)(x)
        x = PReLU()(x)
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = UpSampling2D(size=2)(x)
        x = PReLU()(x)

        x = Conv2D(filters=self.hr_shape[2], kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        return Model(inputs=input_img, outputs=x)

    def train_on_batch(self, lr_batch, hr_batch):

        if len(lr_batch) != len(hr_batch):
            raise ValueError('lr_batch and hr_batch must have the same batch size')

        # -----------------------
        # Train the discriminator
        # -----------------------

        # Train the discriminators to mark real data with 1's
        d_real_loss = self._discriminator.train_on_batch(hr_batch, np.ones(((len(lr_batch),) + (18, 18, 1))))

        # Train the discriminators to mark generated data with 0's
        d_generated_loss = self._discriminator.train_on_batch(self._generator.predict(lr_batch),
                                                              np.zeros(((len(lr_batch),) + (18, 18, 1))))

        d_loss = 0.5 * (d_real_loss + d_generated_loss)

        # -----------------------
        # Train the generator
        # -----------------------

        image_features = self._vgg.predict(hr_batch)
        g_loss = self._combined_model.train_on_batch([lr_batch],
                                              [np.ones(((len(lr_batch),) + (18, 18, 1))), image_features])

        return d_loss, g_loss

    def fit(self, lr_images,
            hr_images,
            epochs=100,
            batch_size=16):

        if len(lr_images) != len(hr_images):
            raise ValueError('lr_images and hr_images must have the same size')

        batches_per_epoch = len(lr_images) // batch_size
        batches_per_epoch += 1 if batches_per_epoch < 1 else 0
        data_iterator = dataIterator(lr_images[50:], hr_images[50:], batch_size)

        iter = 0
        for epoch in range(1, epochs+1):
            tqdm_ = tqdm(range(batches_per_epoch))
            for _ in tqdm_:
                lr_batch, hr_batch = next(data_iterator)
                d_loss, g_loss = self.train_on_batch(lr_batch, hr_batch)
                tqdm_.set_description("EPOCH: {}, D loss: {}, G loss: {}".format(epoch, d_loss, g_loss))

                if iter % 5 == 0:
                    plot_images((lr_images[:50]+1.)*127.5, (self._generator.predict(lr_images[:50])+1.)*127.5,
                                (hr_images[:50]+1.)*127.5, 'images', 1)
                iter += 1


if __name__ == '__main__':
    files = os.listdir(r'C:\Users\Administrator\Desktop\VOCdevkit\processed\low')
    files.sort()
    lr_data = np.array([cv2.imread(r'C:\Users\Administrator\Desktop\VOCdevkit\processed\low\{}'.format(x), 1)
                        / 127.5 - 1. for x in os.listdir(files)])

    files = os.listdir(r'C:\Users\Administrator\Desktop\VOCdevkit\processed\high')
    files.sort()
    hr_data = np.array([cv2.imread(r'C:\Users\Administrator\Desktop\VOCdevkit\processed\high\{}'.format(x), 1)
                        / 127.5 - 1 for x in os.listdir(files)])

    srgan = SRGAN()
    srgan.fit(lr_data, hr_data)
