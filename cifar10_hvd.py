from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"

from tensorflow.keras import datasets, layers, models
from six.moves import cPickle
# import matplotlib.pyplot as plt


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    Arguments:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    Returns:
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


print(tf.__version__)

hvd.init()

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
num_train_samples = 50000

x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
y_train = np.empty((num_train_samples,), dtype='uint8')

for i in range(1, 6):
    fpath = os.path.join('data/cifar10/', 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

# x_test = np.empty((10000, 3, 32, 32), dtype='uint8')
# y_test = np.empty((10000,), dtype='uint8')

# fpath = os.path.join('data/cifar10/', 'test_batch')
# x_test[...], y_test[...] = load_batch(fpath)

train_labels = np.reshape(y_train, (len(y_train), 1))

# test_images = x_test.astype(x_train.dtype)
# test_labels = y_test.astype(y_train.dtype)

# Normalize pixel values to be between 0 and 1
x_train = x_train.transpose(0, 2, 3, 1)
train_images = x_train / 255.0

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(10000 // hvd.size())
train_dataset = train_dataset.shuffle(num_train_samples)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=0.001 * hvd.size())

# checkpoint_dir = './checkpoints'
# checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)

@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = model(images, training=True)
        loss_value = loss(labels, probs)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    return loss_value


for i in range(1000):
    for batch, (images, labels) in enumerate(train_dataset):
        loss_value = training_step(images, labels, batch == 0)

        if batch % 10 == 0 and hvd.local_rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting it.
    # if hvd.rank() == 0:
    #     checkpoint.save(checkpoint_dir)
