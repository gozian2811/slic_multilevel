import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import functools as ft
import os
import shutil
import glob
import math
import random
import time
from config import config
from toolbox import MITools as mt
from toolbox import TensorflowTools as tft

try:
	from tqdm import tqdm  # long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x: x

#constants = bt.read_constants("./constants2.txt")
#MAX_BOUND = float(constants["MAX_BOUND"])
#MIN_BOUND = float(constants["MIN_BOUND"])
#PIXEL_MEAN = float(constants["PIXEL_MEAN"])
MAX_BOUND = float(config['MAX_BOUND'])
MIN_BOUND = float(config['MIN_BOUND'])
PIXEL_MEAN = float(config['PIXEL_MEAN'])
REGION_SIZE = 40
region_half = int(REGION_SIZE/2)

#net construct
bn_params = np.load("models_tensorflow/luna_slh_3D_bndo_flbias_l6_40_aug_stage2/batch_normalization_statistic1.npy")
volume_input = tf.placeholder(tf.float32, [None, REGION_SIZE, REGION_SIZE, REGION_SIZE])
volume_reshape = tf.reshape(volume_input, [-1, REGION_SIZE, REGION_SIZE, REGION_SIZE, 1])
#real_label = tf.placeholder(tf.float32, [None, 2])
net_outs, _, _ = tft.volume_bndo_flbias_l6_40(volume_reshape, dropout_rate=0.0, batch_normalization_statistic=False, bn_params=bn_params)
out_fc2 = net_outs['last_out']
softmax_out = net_outs['sm_out']
#correct_prediction = tf.equal(tf.argmax(softmax_out, 1), tf.argmax(real_label, 1))
#batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	saver.restore(sess, "models_tensorflow/luna_slh_3D_bndo_flbias_l6_40_aug_stage2/epoch28/epoch28")
	input_data = np.load("luna_cubes_56_overbound/subset6/npy_non/1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518_139400_cc_nonannotation.npy")
	ndcenter = np.int_([input_data.shape[0]/2, input_data.shape[1]/2, input_data.shape[2]/2])
	input_data = input_data[ndcenter[0]-region_half:ndcenter[0]+REGION_SIZE-region_half, ndcenter[1]-region_half:ndcenter[1]+REGION_SIZE-region_half, ndcenter[2]-region_half:ndcenter[2]+REGION_SIZE-region_half]
	input_data = np.reshape(input_data, (1, REGION_SIZE, REGION_SIZE, REGION_SIZE))
	input_data[input_data>MAX_BOUND] = MAX_BOUND
	input_data[input_data<MIN_BOUND] = MIN_BOUND
	input_data = (input_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
	predictions = sess.run(softmax_out, {volume_input: input_data})
	print predictions