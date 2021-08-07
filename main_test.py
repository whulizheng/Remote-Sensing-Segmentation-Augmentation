import tensorflow as tf
from GANs_models import pix2pix
import Utils

BUFFER_SIZE = 400
EPOCHS = 100
BATCH_SIZE = 1
shape = [256, 256]
test_img_path = "test.png"
# 加载测试图片并预处理
image = tf.io.read_file(test_img_path)
image = tf.image.decode_jpeg(image)
image = tf.image.resize(
    image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
image = tf.cast(image, tf.float32)
input_image = (image / 127.5) - 1
input_image = input_image[:, :, 0:3]
input_image = tf.reshape(input_image, [1, 256, 256, 3])

# 加载模型
pix2pix = pix2pix.pix2pix()
pix2pix.load_model()

# 测试单张图片
pix2pix.generate_images(input_image, tag="output")
