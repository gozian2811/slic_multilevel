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

SNAPSHOT_EPOCH = 1

REGION_SIZE = 40
BATCH_SIZE = 10
VALIDATION_RATE = 0.2
NEGATIVE_VALIDATION = True

SCALE_AUGMENTATION = False
TRANSLATION_AUGMENTATION = False
ROTATION_AUGMENTATION = False
FLIP_AUGMENTATION = False
CENTERING = True
AUGMENTATION = SCALE_AUGMENTATION or TRANSLATION_AUGMENTATION or ROTATION_AUGMENTATION or FLIP_AUGMENTATION

print("batch size:{}" .format(BATCH_SIZE))
print("region size:{}" .format(REGION_SIZE))
print("max bound:{}" .format(MAX_BOUND))
print("min bound:{}" .format(MIN_BOUND))
print("pixel mean:{}" .format(PIXEL_MEAN))
region_half = int(REGION_SIZE/2)

load_path = "models_tensorflow"

net_init_name = "luna_slh_3D_bndo_flbias_l6_40_aug_stage2"
net_init_path = load_path + "/" + net_init_name
net_init_file = net_init_path + "/epoch28/epoch28"
pfilelist_path = net_init_path + "/pfilelist.log"
nfilelist_path = net_init_path + "/nfilelist.log"
vision_path = "detection_vision"

print("read pfilelist from: %s" %(pfilelist_path))
pfilelist_file = open(pfilelist_path, "r")
pfiles = pfilelist_file.readlines()
for pfi in range(len(pfiles)):
	pfiles[pfi] = pfiles[pfi][:-1]
pfilelist_file.close()

print("read nfilelist from: %s" % (nfilelist_path))
nfilelist_file = open(nfilelist_path, "r")
nfiles = nfilelist_file.readlines()
for nfi in range(len(nfiles)):
	nfiles[nfi] = nfiles[nfi][:-1]
nfilelist_file.close()

num_positive = len(pfiles)
num_negative = len(nfiles)

positive_val_num = int(num_positive * VALIDATION_RATE)
positive_train_num = num_positive - positive_val_num
negative_val_num = int(num_negative * VALIDATION_RATE)
negative_train_num = num_negative - negative_val_num
tpfiles = pfiles[:positive_train_num]
vpfiles = pfiles[positive_train_num:num_positive]
positive_train_indices = [i for i in range(positive_train_num)]
positive_val_indices = [i for i in range(positive_val_num)]
tnfiles = nfiles[:negative_train_num]
vnfiles = nfiles[negative_train_num:num_negative]
negative_train_indices = [i for i in range(negative_train_num)]
negative_val_indices = [i for i in range(negative_val_num)]

#random.shuffle(positive_train_indices)
#random.shuffle(positive_val_indices)

#net construct
bn_params = np.load(net_init_path + "/batch_normalization_statistic1.npy")
volume_input = tf.placeholder(tf.float32, [None, REGION_SIZE, REGION_SIZE, REGION_SIZE])
volume_reshape = tf.reshape(volume_input, [-1, REGION_SIZE, REGION_SIZE, REGION_SIZE, 1])
real_label = tf.placeholder(tf.float32, [None, 2])
#r_bn1, b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, softmax_out = tft.volume_bnnet2_l6_56(volume_reshape)
net_outs, _, _ = tft.volume_bndo_flbias_l6_40(volume_reshape, dropout_rate=0.0, batch_normalization_statistic=False, bn_params=bn_params)
out_fc2 = net_outs['last_out']
softmax_out = net_outs['sm_out']
correct_prediction = tf.equal(tf.argmax(softmax_out, 1), tf.argmax(real_label, 1))
batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

extract_volumes = ft.partial(mt.extract_volumes, volume_shape=np.int_([REGION_SIZE, REGION_SIZE, REGION_SIZE]), centering=CENTERING, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)
if not AUGMENTATION:
	correct_output = open(vision_path + "/correct_predictions2.log", "w")
	incorrect_output = open(vision_path + "/incorrect_predictions2.log", "w")

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	print("loading net from file:{}" .format(net_init_file))
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	saver.restore(sess, net_init_file)
	
	ptd_num = 0
	ptd_accuracy = 0.0
	if not AUGMENTATION:
		correct_output.write("positive training:\n")
		incorrect_output.write("positive training:\n")
	for pbi in tqdm(range(0, positive_train_num, BATCH_SIZE)):
		posbatchend = min(pbi+BATCH_SIZE, positive_train_num)
		for pti in range(pbi, posbatchend):
			data_index = positive_train_indices[pti]
			pfile = tpfiles[data_index]
			positive_data = np.load(pfile)
			nodule_diameter = 0
			if "positive_batch" not in dir():
				positive_batch = extract_volumes(positive_data, nodule_diameter=nodule_diameter)
			else:
				positive_batch = np.concatenate((positive_batch, extract_volumes(positive_data, nodule_diameter=nodule_diameter)), axis=0)
		positive_batch = mt.medical_normalization(positive_batch, MAX_BOUND, MIN_BOUND, PIXEL_MEAN, True)
		positive_label = np.zeros(shape=(positive_batch.shape[0], 2), dtype=float)
		positive_label[:,0] = 1
		predictions, accuracies = sess.run([out_fc2, correct_prediction], {volume_input: positive_batch, real_label: positive_label})
		if not AUGMENTATION:
			for a in range(len(accuracies)):
				data_index = positive_train_indices[pbi+a]
				pfile = tpfiles[data_index]
				if accuracies[a]:
					correct_output.write(pfile+" {}\n" .format(predictions[a]))
				else:
					incorrect_output.write(pfile+" {}\n" .format(predictions[a]))
		for di in range(positive_batch.shape[0]):
			ptd_num += 1
			ptd_accuracy += accuracies[di]
		
		del positive_batch

	if ptd_num > 0:
		ptd_accuracy /= ptd_num
	print("positive training accuracy:%f" %(ptd_accuracy))
	
	pvd_num = 0
	pvd_accuracy = 0.0
	if not AUGMENTATION:
		correct_output.write("\npositive validation:\n")
		incorrect_output.write("\npositive validation:\n")
	for pbi in tqdm(range(0, positive_val_num, BATCH_SIZE)):
		posbatchend = min(pbi+BATCH_SIZE, positive_val_num)
		for pvi in range(pbi, posbatchend):
			data_index = positive_val_indices[pvi]
			pfile = vpfiles[data_index]
			positive_data = np.load(pfile)
			nodule_diameter = 0
			if "positive_batch" not in dir():
				positive_batch = extract_volumes(positive_data, nodule_diameter=nodule_diameter)
			else:
				positive_batch = np.concatenate((positive_batch, extract_volumes(positive_data, nodule_diameter=nodule_diameter)), axis=0)
		positive_batch = mt.medical_normalization(positive_batch, MAX_BOUND, MIN_BOUND, PIXEL_MEAN, True)
		positive_label = np.zeros(shape=(positive_batch.shape[0], 2), dtype=float)
		positive_label[:,0] = 1
		predictions, accuracies = sess.run([out_fc2, correct_prediction], {volume_input: positive_batch, real_label: positive_label})
		if not AUGMENTATION:
			for a in range(len(accuracies)):
				data_index = positive_val_indices[pbi+a]
				pfile = vpfiles[data_index]
				if accuracies[a]:
					correct_output.write(pfile+" {}\n" .format(predictions[a]))
				else:
					incorrect_output.write(pfile+" {}\n" .format(predictions[a]))
		for di in range(positive_batch.shape[0]):
			pvd_num += 1
			pvd_accuracy += accuracies[di]
		
		del positive_batch

	if pvd_num > 0:
		pvd_accuracy /= pvd_num
	print("positive validation accuracy:%f" %(pvd_accuracy))
	
	if NEGATIVE_VALIDATION:
		ntd_num = 0
		ntd_accuracy = 0.0	
		if not AUGMENTATION:
			correct_output.write("\nnegative training:\n")
			incorrect_output.write("\nnegative training:\n")
		for nbi in tqdm(range(0, negative_train_num, BATCH_SIZE)):
			negbatchend = min(nbi+BATCH_SIZE, negative_train_num)
			negative_batch = np.empty((negbatchend-nbi, REGION_SIZE, REGION_SIZE, REGION_SIZE), dtype=float)
			for nti in range(nbi, negbatchend):
				data_index = negative_train_indices[nti]
				nfile = tnfiles[data_index]
				negative_data = np.load(nfile)
				ndcenter = np.int_([negative_data.shape[0]/2, negative_data.shape[1]/2, negative_data.shape[2]/2])
				negative_batch[nti-nbi] = negative_data[ndcenter[0]-region_half:ndcenter[0]+REGION_SIZE-region_half, ndcenter[1]-region_half:ndcenter[1]+REGION_SIZE-region_half, ndcenter[2]-region_half:ndcenter[2]+REGION_SIZE-region_half]
			negative_batch = mt.medical_normalization(negative_batch, MAX_BOUND, MIN_BOUND, PIXEL_MEAN, True)
			negative_label = np.zeros(shape=(negative_batch.shape[0], 2), dtype=float)
			negative_label[:,1] = 1
			predictions, accuracies = sess.run([out_fc2, correct_prediction], {volume_input: negative_batch, real_label: negative_label})
			if not AUGMENTATION:
				for a in range(len(accuracies)):
					data_index = negative_train_indices[nbi+a]
					nfile = tnfiles[data_index]
					if accuracies[a]:
						correct_output.write(nfile+" {}\n" .format(predictions[a]))
					else:
						incorrect_output.write(nfile+" {}\n" .format(predictions[a]))
			for di in range(negative_batch.shape[0]):
				ntd_num += 1
				ntd_accuracy += accuracies[di]
			
			del negative_batch

		if ntd_num > 0:
			ntd_accuracy /= ntd_num
		print("negative training accuracy:%f" %(ntd_accuracy))
		
		nvd_num = 0
		nvd_accuracy = 0.0
		if not AUGMENTATION:
			correct_output.write("\nnegative validation:\n")
			incorrect_output.write("\nnegative validation:\n")
		for nbi in tqdm(range(0, negative_val_num, BATCH_SIZE)):
			negbatchend = min(nbi+BATCH_SIZE, negative_val_num)
			negative_batch = np.empty((negbatchend-nbi, REGION_SIZE, REGION_SIZE, REGION_SIZE), dtype=float)
			for nvi in range(nbi, negbatchend):
				data_index = negative_val_indices[nvi]
				nfile = vnfiles[data_index]
				negative_data = np.load(nfile)
				ndcenter = np.int_([negative_data.shape[0]/2, negative_data.shape[1]/2, negative_data.shape[2]/2])
				negative_batch[nvi-nbi] = negative_data[ndcenter[0]-region_half:ndcenter[0]+REGION_SIZE-region_half, ndcenter[1]-region_half:ndcenter[1]+REGION_SIZE-region_half, ndcenter[2]-region_half:ndcenter[2]+REGION_SIZE-region_half]
			negative_batch = mt.medical_normalization(negative_batch, MAX_BOUND, MIN_BOUND, PIXEL_MEAN, True)
			negative_label = np.zeros(shape=(negative_batch.shape[0], 2), dtype=float)
			negative_label[:,1] = 1
			predictions, accuracies = sess.run([out_fc2, correct_prediction], {volume_input: negative_batch, real_label: negative_label})
			if not AUGMENTATION:
				for a in range(len(accuracies)):
					data_index = negative_val_indices[nbi+a]
					nfile = vnfiles[data_index]
					if accuracies[a]:
						correct_output.write(nfile+" {}\n" .format(predictions[a]))
					else:
						incorrect_output.write(nfile+" {}\n" .format(predictions[a]))
			for di in range(negative_batch.shape[0]):
				nvd_num += 1
				nvd_accuracy += accuracies[di]
			
			del negative_batch

		if nvd_num > 0:
			nvd_accuracy /= nvd_num
		print("negative validation accuracy:%f" %(nvd_accuracy))

	#print("positive_train_acc:%f positive_val_acc:%f" %(ptd_accuracy, pvd_accuracy))
	#print("positive_train_acc:%f negative_train_acc:%f\npositive_val_acc:%f negative_val_acc:%f" %(ptd_accuracy, ntd_accuracy, pvd_accuracy, nvd_accuracy))

if not AUGMENTATION:
	correct_output.close()
	incorrect_output.close()