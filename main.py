from __future__ import print_function, division

import os
import imageio
import datetime
import argparse
import numpy as np  

from tqdm import tqdm

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras import backend as K


class MemNet:
    def __init__(self):
        self.memsize = 50000
        self.latent_dim = 128
        self.imsize = 28 
        self.imchannels = 1
        self.img_shape = (self.imsize, self.imsize, self.imchannels)

        optimizer = Adam(0.0002, 0.5)

        # Load the dataset
        (X_train, self.y_train), (_, _) = mnist.load_data()
        X_train = np.expand_dims(X_train, axis=3)
 
        # Rescale -1 to 1
        self.X_train = X_train / 127.5 - 1.

        # Calculate output shape of D (PatchGAN)
        patch = int(self.imsize / 2**2)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of D
        self.df = 64

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.discriminator.summary()

        self.generator = self.build_network()

        label = Input(shape=(1,), dtype='int32')

        fake = self.generator(label)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images 
        valid = self.discriminator(fake)

        self.model = Model(inputs=label, outputs=[valid, fake])
        self.model.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_network(self):
      
        model = self.build_generator()
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.memsize, self.latent_dim)(label))
        image = model(label_embedding)

        return Model(label, image)

    def build_generator(self):
      
        model = Sequential()

        factor = 7

        model.add(Dense(factor*factor*128, input_dim=self.latent_dim))
        model.add(Reshape((factor, factor, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.imchannels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        return model

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d2)

        return Model(img, validity)
    
    def train(self, epochs, batch_size=400, save_interval=50):

        start_epoch = 0
        start_time = datetime.datetime.now()

        n_batches = self.memsize // batch_size

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake  = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(start_epoch, epochs):
            for batch in range(n_batches):

                idxs = np.arange(batch*batch_size, (batch+1)*batch_size)
                imgs = self.X_train[idxs]


                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Generate fake images
                fake_imgs = self.generator.predict(idxs)

                # Train the discriminator (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.model.train_on_batch(idxs, [valid, imgs])
    
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch+1, n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[1],
                                                                        elapsed_time))
            # If at save interval => save generated image samples
            if (epoch+1) % save_interval == 0:
                self.generator.save("models/model-{}.h5".format(epoch+1))

    def latent_interpolation(self, model_path, out_path, n_steps, start_num, end_num):
        
        full_model = load_model(model_path)
        embeddings = full_model.layers[1].get_weights()[0]

        generator = full_model.layers[-1]
        model = Model(generator.layers[0].input, generator.layers[-1].output)

        num = start_num
        countdown = start_num > end_num

        vectors = []
        alpha_values = np.linspace(0, 1, n_steps)

        start_idx = np.random.choice(self.y_train==num)

        for _ in tqdm(range(abs(start_num-end_num))):
            latent_start = embeddings[start_idx]
            num = num - 1 if countdown else num + 1
            start_idx = np.random.choice(self.y_train==num)
            latent_end = embeddings[start_idx] 
            for alpha in alpha_values:
                vector = latent_start*(1-alpha) + latent_end*alpha
                vectors.append(vector)

        gen_imgs = model.predict(np.array(vectors))
        gen_imgs = np.uint8((0.5 * gen_imgs + 0.5) * 255)
        imageio.mimsave(os.path.join(out_path, 'latent.gif'), gen_imgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Counter Paramaters')
    # training parameters
    parser.add_argument('-t', '--train',         action="store_true", default=False)
    parser.add_argument('-e', '--epochs',        type=int, default=1000, help='number of epochs for training')
    parser.add_argument('-b', '--batch-size',    type=int, default=400, help='batch size used in training')
    parser.add_argument('-s', '--save-interval', type=int, default=50, help='save model every interval')
    # generation parameters
    parser.add_argument('-g', '--generate',  action="store_true", default=True)
    parser.add_argument('-a', '--start-num', type=int, default=9, help='count down/up starts with this number')
    parser.add_argument('-z', '--end-num',   type=int, default=0, help='count down/up ends with this number')
    parser.add_argument('-n', '--n-steps',   type=int, default=40, help='number of steps used in latent interpolation')
    parser.add_argument('-o', '--out-path',  type=str, default="./", help='path to generated gif')
    parser.add_argument('-m', '--model-path',type=str, default="models/model-1000.h5", help='path to saved model')
    args = parser.parse_args()

    memnet = MemNet()
    if args.train:
        memnet.train(epochs=args.epochs, batch_size=args.batch_size, save_interval=args.save_interval)
    if args.generate:
        memnet.latent_interpolation(model_path=args.model_path, out_path=args.out_path, 
            n_steps=args.n_steps, start_num=args.start_num, end_num=args.end_num)