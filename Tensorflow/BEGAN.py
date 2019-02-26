import numpy as np
import tensorflow as tf
from utils import dataIterator, sample_images
from tqdm import tqdm
from keras.datasets import mnist


class BEGAN:

    def __init__(self, X_input_tensor,
                       Z_input_tensor,
                       generator_latent_dim=10,
                       lambda_term=0.001,
                       gamma=2000,
                       pow=1,
                       discriminator_encoder_layers=[100, 10],
                       discriminator_decoder_layers=[100, 28**2],
                       generator_layers=[100, 28**2],
                       discriminator_optimizer=tf.train.AdamOptimizer(0.001),
                       generator_optimizer=tf.train.AdamOptimizer(0.001),
                       discriminator_encoder_hidden_activation=tf.nn.relu,
                       discriminator_decoder_hidden_activation=tf.nn.relu,
                       discriminator_encoder_output_activation=tf.nn.relu,
                       discriminator_decoder_output_activation=tf.nn.tanh,
                       generator_hidden_activation=tf.nn.elu,
                       generator_output_activation=tf.nn.tanh):

        self.X_input_tensor = X_input_tensor
        self.Z_input_tensor = Z_input_tensor
        self.generator_latent_dim = generator_latent_dim
        self.k_t = 0.
        self.discriminator_encoder_layers = discriminator_encoder_layers
        self.discriminator_decoder_layers = discriminator_decoder_layers
        self.generator_layers = generator_layers
        self.discriminator_encoder_hidden_activation = discriminator_encoder_hidden_activation
        self.discriminator_decoder_hidden_activation = discriminator_decoder_hidden_activation
        self.discriminator_encoder_output_activation = discriminator_encoder_output_activation
        self.discriminator_decoder_output_activation = discriminator_decoder_output_activation
        self.generator_hidden_activation = generator_hidden_activation
        self.generator_output_activation = generator_output_activation

        # GAN
        self.g_out = self.build_generator(self.Z_input_tensor)
        self.d_generated_out = self.build_discriminator(self.g_out)
        self.d_real_out = self.build_discriminator(self.X_input_tensor, reuse=True)

        # Loss and k update
        loss_d_generated = BEGAN.L_pow_Norm(self.g_out, self.d_generated_out, pow=pow)
        loss_d_real = BEGAN.L_pow_Norm(self.X_input_tensor, self.d_real_out, pow=pow)
        self.D_loss = loss_d_real - self.k_t * loss_d_generated
        self.G_loss = loss_d_generated
        self.k_t_new = tf.add(self.k_t, lambda_term * (gamma * loss_d_real - loss_d_generated))

        # compile
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
        self.gen_step = generator_optimizer.minimize(self.G_loss, var_list=gen_vars)  # G Train step
        self.disc_step = discriminator_optimizer.minimize(self.D_loss, var_list=disc_vars)  # D Train step

        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)


    @staticmethod
    def L_pow_Norm(y_true,
                   y_pred,
                   pow=2):
        diff = tf.abs(y_pred - y_true)
        return tf.reduce_mean(tf.pow(diff, pow))

    def build_generator(self, input_tensor,
                        reuse=False,
                        scope_name='generator'):
        with tf.variable_scope(scope_name, reuse=reuse):
            x = input_tensor
            for layer_dim in self.generator_layers[:-1]:
                x = tf.layers.dense(x, units=layer_dim, activation=self.generator_hidden_activation)
            x = tf.layers.dense(x, units=self.generator_layers[-1], activation=self.generator_output_activation)
            return x

    def build_discriminator(self, input_tensor, reuse=False, scope_name='discriminator', name='discriminator'):
        with tf.variable_scope(scope_name, reuse=reuse):
            # ENCODE
            x = input_tensor
            for i, layer_dim in enumerate(self.discriminator_encoder_layers[:-1]):
                x = tf.layers.dense(x, units=layer_dim, activation=self.discriminator_encoder_hidden_activation,
                                    name='encode_{}_{}'.format(name, i))
            x = tf.layers.dense(x,
                                units=self.discriminator_encoder_layers[-1],
                                activation=self.discriminator_encoder_output_activation,
                                name='encode_{}_last'.format(name))

            # DECODE
            for i, layer_dim in enumerate(self.discriminator_decoder_layers[:-1]):
                x = tf.layers.dense(x,
                                    units=layer_dim,
                                    activation=self.discriminator_decoder_hidden_activation,
                                    name='decode_{}_{}'.format(name, i))
            x = tf.layers.dense(x,
                                units=self.discriminator_decoder_layers[-1],
                                activation=self.discriminator_decoder_output_activation,
                                name='decode_{}_last'.format(name))
            return x

    def train_on_batch(self, x,
                       z):
        if len(x) != len(z):
            raise ValueError('x and z must have the same batch size')

        _, _, dloss, gloss, k_t_new = self.sess.run(
                [self.disc_step, self.gen_step, self.D_loss, self.G_loss, self.k_t_new],
                feed_dict={self.X_input_tensor: x, self.Z_input_tensor: z,}
        )
        self.k_t = np.clip(k_t_new, 0, 1)

        return dloss, gloss, self.k_t

    def fit(self, x,
            batch_size=8,
            epochs=1,
            verbose=True):

        data_iterator = dataIterator(x, batch_size)
        batches_per_epoch = len(x) // batch_size

        batches_per_epoch += 1 if batches_per_epoch < 1 else 0

        for epoch in range(1, epochs + 1):
            tqdm_ = tqdm(range(batches_per_epoch), disable=not verbose)
            for _ in tqdm_:
                x_current_batch = next(data_iterator)
                z = np.random.normal(0, 1, (batch_size, self.generator_latent_dim))
                d_loss, g_loss, k_t = self.train_on_batch(x_current_batch, z)
                tqdm_.set_description("EPOCH: {}, D loss: {}, G loss: {}, k: {}".format(epoch, d_loss, g_loss, k_t))

            if verbose:
                idxs = np.arange(0, len(x))
                np.random.shuffle(idxs)
                idxs = idxs[:25]

                reconstructed_images = self.sess.run([self.d_real_out], {self.X_input_tensor: x[idxs]})[0]
                reconstructed_images = (1 + reconstructed_images) * 127.5
                sample_images(reconstructed_images.reshape((25, 28, 28, 1)), '{}_real'.format(epoch), 'images')
                generated_images = (1 + self.generate(25)) * 127.5
                sample_images(generated_images.reshape((25, 28, 28, 1)), '{}_generated'.format(epoch), 'images')


    def generate(self, quantity=25):
        z = np.random.normal(0, 1, (quantity, self.generator_latent_dim))
        generated_data = self.sess.run([self.g_out], {self.Z_input_tensor: z})
        return generated_data[0]


if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = X_train.reshape((len(X_train), 784))

    generator_latent_dim = 10

    X = tf.placeholder(tf.float32, [None, 784])
    Z = tf.placeholder(tf.float32, [None, generator_latent_dim])

    gan = BEGAN(X, Z, generator_latent_dim)
    gan.fit(X_train, epochs=30)
