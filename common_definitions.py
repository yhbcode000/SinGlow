import tensorflow as tf
import math
import numpy as np

LOAD_WEIGHT = True

# model parameter
SQUEEZE_FACTOR = 2
K_GLOW = 12
L_GLOW = 8
WINDOW_SIZE = 1 # in seconds
SAMPLING_RATE = 16384 # target sampling rate
WINDOW_LENGTH = int(WINDOW_SIZE*SAMPLING_RATE) # better to be mult of SQUEEZE_FACTOR
CHANNEL_SIZE = 1
BUFFER_SIZE = 20000
ACTIVATION = tf.nn.relu6
KERNEL_INITIALIZER_CLOSE_VALUE = lambda x=0: tf.random_normal_initializer(x, 1e-4)
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()

# training parameters
LEARNING_RATE = 1e-3
LAMBDA_LIPSCHITZ = 1e-3
DROPOUT_N = 0.1
BATCH_SIZE = 32
EPOCHS = 50
CHECKPOINT_PATH = "./checkpoints/weights"
TENSORBOARD_LOGDIR = "./logs/GLOW"

# dataset parameters
ALPHA_BOUNDARY = 0.05

# general
TF_EPS = tf.keras.backend.epsilon()
TEMP = 1