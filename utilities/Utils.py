import tensorflow as tf
from utilities import Constants
import os


def load(image_file,reverse = 0):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    if reverse:
        return real_image,input_image
    else:
        return input_image,real_image



def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image, shape):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, shape[0], shape[1], 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def random_mirror(input_image, real_image):
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
        return input_image, real_image
    else:
        return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image, shape):

    input_image, real_image = resize(
        input_image, real_image, shape[0], shape[1])
    input_image, real_image = random_crop(input_image, real_image, shape)
    input_image, real_image = random_mirror(input_image, real_image)
    return input_image, real_image


def load_image_train(image_file, shape,reverse=0):
    input_image, real_image = load(image_file,reverse)
    input_image, real_image = random_jitter(input_image, real_image, shape)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file, shape,reverse=0):
    input_image, real_image = load(image_file,reverse)
    input_image, real_image = resize(
        input_image, real_image, shape[0], shape[1])
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_model(model_name, output_directory):
    if model_name == 'cGAN':
        from models import cGAN
        return cGAN.cGAN(output_directory)
    if model_name == 'cycleGAN':
        from models import cycleGAN
        return cycleGAN.cycleGAN(output_directory)
    if model_name == 'dcGAN':
        from models import dcGAN
        return dcGAN.dcGAN(output_directory)
    if model_name == 'pix2pix':
        from models import pix2pix
        return pix2pix.pix2pix(output_directory)
    raise ValueError(model_name + ' is not the supported MODEL!')
