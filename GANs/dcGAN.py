from matplotlib import pyplot as plt
from IPython import display
import datetime
import tensorflow as tf
from tensorflow.keras import layers
import os
import time
import pandas
OUTPUT_CHANNELS = 3


class dcGAN():
    def __init__(self):
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.checkpoint_dir = 'training_checkpoints/training_checkpoints_dcGAN'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.log_dir = "logs/loss/dcGAN.csv"

    def Generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64*64*64, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((64, 64, 64)))
        assert model.output_shape == (None, 64, 64, 64)  # 注意：batch size 没有限制

        model.add(layers.Conv2DTranspose(
            32, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(
            16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 128, 128, 16)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                  padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 256, 256, 3)

        return model

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def Discriminator(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[256, 256, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generate_images(self, tar=None, tag="tmp"):
        noise = tf.random.normal([1, 100])
        prediction = self.generator(noise, training=True)
        plt.figure(figsize=(15, 15))
        if tar != None:
            display_list = [tar[0], prediction[0]]
            title = ['Ground Truth', 'Generated Image']

            for i in range(2):
                plt.subplot(1, 2, i+1)
                plt.title(title[i])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
            plt.savefig(tag+".png")
            plt.clf()
            plt.close('all')
        else:
            display_list = [prediction[0]]
            title = ['Generated Image']

            for i in range(1):
                plt.subplot(1, 1, i+1)
                plt.title(title[i])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
            plt.savefig(tag+".png")
            plt.clf()
            plt.close('all')
    def save_loss_log(self,gen,dis,path):
        df = pandas.DataFrame([gen,dis],index=["gen","disc"])
        df.to_csv(path,header=False)
    @tf.function
    def train_step(self, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            noise = tf.random.normal([len(target), 100])
            gen_output = self.generator(noise, training=True)

            disc_real_output = self.discriminator(
                target, training=True)
            disc_generated_output = self.discriminator(
                gen_output, training=True)

            gen_loss = self.generator_loss(
                disc_generated_output)
            disc_loss = self.discriminator_loss(
                disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        return gen_loss,disc_loss

    def fit(self, train_ds, epochs, test_ds):
        gen_losses = []
        disc_losses = []
        for epoch in range(epochs):
            start = time.time()
            display.clear_output(wait=True)
            # self.generate_images(tag = "tmp/"+str(epoch)+"_tmp")
            print("Epoch: ", epoch)

            # Train
            for n, (input_image, target) in train_ds.enumerate():
                print('.', end='')
                if (n+1) % 100 == 0:
                    print()
                gen_loss,disc_loss = self.train_step(target, epoch)
                gen_losses.append(float(gen_loss))
                disc_losses.append(float(disc_loss))
            print()

            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time()-start))
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        self.save_loss_log(gen_losses,disc_losses,self.log_dir)

    def load_model(self):
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))
