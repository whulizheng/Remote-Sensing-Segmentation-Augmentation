from matplotlib import pyplot as plt
from IPython.display import clear_output
import datetime
import tensorflow as tf
import pandas
import os
import time
OUTPUT_CHANNELS = 3


class cycleGAN():
    def __init__(self) -> None:
        self.LAMBDA = 10
        self.checkpoint_dir = "training_checkpoints/training_checkpoints_cycleGAN"
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_g = self.Generator()
        self.generator_f = self.Generator()
        self.discriminator_x = self.Discriminator()
        self.discriminator_y = self.Discriminator()
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoints = tf.train.Checkpoint(generator_g=self.generator_g,
                                             generator_f=self.generator_f,
                                             discriminator_x=self.discriminator_x,
                                             discriminator_y=self.discriminator_y,
                                             generator_g_optimizer=self.generator_g_optimizer,
                                             generator_f_optimizer=self.generator_f_optimizer,
                                             discriminator_x_optimizer=self.discriminator_x_optimizer,
                                             discriminator_y_optimizer=self.discriminator_y_optimizer)
        self.log_dir = "logs/loss/cycleGAN.csv"


    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2,
                   padding='same', kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
            # (bs, 128, 128, 64)
            self.downsample(64, 4, apply_batchnorm=False),
            self.downsample(128, 4),  # (bs, 64, 64, 128)
            self.downsample(256, 4),  # (bs, 32, 32, 256)
            self.downsample(512, 4),  # (bs, 16, 16, 512)
            self.downsample(512, 4),  # (bs, 8, 8, 512)
            self.downsample(512, 4),  # (bs, 4, 4, 512)
            self.downsample(512, 4),  # (bs, 2, 2, 512)
            self.downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            self.upsample(512, 4),  # (bs, 16, 16, 1024)
            self.upsample(256, 4),  # (bs, 32, 32, 512)
            self.upsample(128, 4),  # (bs, 64, 64, 256)
            self.upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')

        x = inp

        down1 = self.downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
            zero_pad2)  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=inp, outputs=last)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)

        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss
    def save_loss_log(self,gen,dis,path):
        df = pandas.DataFrame([gen,dis],index=["gen","disc"])
        df.to_csv(path,header=False)
    def generate_images(self, test_input, tag="tmp", if_g_g=1):
        prediction = None
        if if_g_g:
            prediction = self.generator_g(test_input)
        else:
            prediction = self.generator_f(test_input)

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Generated Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # 获取范围在 [0, 1] 之间的像素值以绘制它。
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        # plt.show()
        plt.savefig(tag+".png")
        plt.clf()
        plt.close('all')

    @tf.function
    def train_step(self, real_x, real_y):
        # persistent 设置为 Ture，因为 GradientTape 被多次应用于计算梯度。
        with tf.GradientTape(persistent=True) as tape:
            # 生成器 G 转换 X -> Y。
            # 生成器 F 转换 Y -> X。

            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x 和 same_y 用于一致性损失。
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # 计算损失。
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(
                real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # 总生成器损失 = 对抗性损失 + 循环损失。
            total_gen_g_loss = gen_g_loss + \
                total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + \
                total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # 计算生成器和判别器损失。
        generator_g_gradients = tape.gradient(total_gen_g_loss,
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss,
                                              self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  self.discriminator_y.trainable_variables)

        # 将梯度应用于优化器。
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))

        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))
        return gen_g_loss, disc_x_loss

    def fit(self, train_ds, epochs, test_ds):
        gen_losses = []
        disc_losses = []
        for epoch in range(epochs):
            clear_output(wait=True)
            '''
            for example_input, example_target in test_ds.take(1):
                self.generate_images(
                    example_input, "tmp/"+str(epoch)+"_tmp")
            '''
            print("Epoch: ", epoch)
            start = time.time()
            gen_g_loss = 0
            disc_x_loss = 0
            for n, (input_image, target) in train_ds.enumerate():
                gen_g_loss, disc_x_loss = self.train_step(input_image, target)
                if n % 10 == 0:
                    print('.', end='')
            gen_losses.append(float(gen_g_loss))
            disc_losses.append(float(disc_x_loss))
            # saving (checkpoint) the model every 20 epochs
            '''
            if (epoch + 1) % 20 == 0:
                self.checkpoints.save(file_prefix=self.checkpoint_prefix)
            '''
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time()-start))
        self.checkpoints.save(file_prefix=self.checkpoint_prefix)
        self.save_loss_log(gen_losses,disc_losses,self.log_dir)

    def load_model(self):
        self.checkpoints.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))