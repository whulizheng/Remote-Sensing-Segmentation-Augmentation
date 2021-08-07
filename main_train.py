import tensorflow as tf
from GANs import cGAN
import Utils

BUFFER_SIZE = 400
EPOCHS = 100
BATCH_SIZE = 1
shape = [256, 256]


train_PATH = "data/HRSC2016/training/"
test_PATH = "data/HRSC2016/test/"

# 加载数据集
train_dataset = tf.data.Dataset.list_files(train_PATH+'*.png')
train_dataset = train_dataset.map(lambda x: Utils.load_image_train(x, shape))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(test_PATH+'*.png')
test_dataset = test_dataset.map(lambda x: Utils.load_image_train(x, shape))
test_dataset = test_dataset.batch(BATCH_SIZE)

# 初始化模型
GAN = cGAN.cGAN()
# 加载训练那好的模型
# GAN.load_model()
# 训练模型
GAN.fit(train_dataset, EPOCHS, test_dataset)
