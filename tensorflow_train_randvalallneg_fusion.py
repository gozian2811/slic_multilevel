#modification requirements:focal loss modulation, medical normalization.

import tensorflow as tf
import numpy as np
import functools as ft
import os
import shutil
import glob
import math
import random
import time
#from tensorflow.python import debug as tf_debug
from toolbox.config import config
from toolbox import BasicTools as bt
from toolbox import MITools as mt
from toolbox import TensorflowTools as tft

try:
	from tqdm import tqdm  # long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x: x

MAX_BOUND = float(config["MAX_BOUND"])
MIN_BOUND = float(config["MIN_BOUND"])
PIXEL_MEAN = float(config["PIXEL_MEAN"])
NORM_CROP = config["NORM_CROP"]

#NUM_EPOCH = 200
SNAPSHOT_EPOCH = 1
#DECAY_EPOCH = 0
#INITIAL_LEARNING_RATE = 0.001
#DECAY_LEARNING_RATE = 1.0

ALL_NEGATIVE = False
REGION_SIZES = [20, 30, 40]
BATCH_SIZE = 15
VALIDATION_RATE = 0.2

SCALE_AUGMENTATION = False
TRANSLATION_AUGMENTATION = False
ROTATION_AUGMENTATION = True
FLIP_AUGMENTATION = True
CENTERING = True
AUGMENTATION = SCALE_AUGMENTATION or TRANSLATION_AUGMENTATION or ROTATION_AUGMENTATION or FLIP_AUGMENTATION

print("batch size:{}" .format(BATCH_SIZE))
print("region sizes:{}" .format(REGION_SIZES))
print("max bound:{}" .format(MAX_BOUND))
print("min bound:{}" .format(MIN_BOUND))
print("pixel mean:{}" .format(PIXEL_MEAN))
region_halfs = [int(REGION_SIZES[0]/2), int(REGION_SIZES[1]/2), int(REGION_SIZES[2]/2)]

store_path = "models_tensorflow"
load_path = "models_tensorflow"
net_store_name = "luna_slh_3D_fusion_test"
net_store_path = store_path + "/" + net_store_name
tensorboard_path = net_store_path + "/tensorboard/"
net_init_names = ["luna_slh_3D_bndo_flbias_l5_20_aug_stage3", "luna_slh_3D_bndo_flbias_l5_30_aug2_stage2", "luna_slh_3D_bndo_flbias_l6_40_aug_stage2"]
if 'net_init_names' in dir():
	net_init_paths = [load_path + "/" + net_init_names[0], load_path + "/" + net_init_names[1], load_path + "/" + net_init_names[2]]
	net_init_files = [net_init_paths[0]+"/epoch20/epoch20", net_init_paths[1]+"/epoch10/epoch10", net_init_paths[2]+"/epoch28/epoch28", "models_tensorflow/luna_slh_3D_fusion/epoch6/epoch6"]
	#pfilelist_path = net_init_path + "/pfilelist.log"
	#nfilelist_path = net_init_path + "/nfilelist.log"

if os.access(net_store_path, os.F_OK):
	shutil.rmtree(net_store_path)
os.makedirs(net_store_path)
if os.access(tensorboard_path, os.F_OK):
	shutil.rmtree(tensorboard_path)

#data arrangement
data_dir1 = "luna_cubes_56_overbound"
#data_dir2 = "tianchi_cubes_56_overbound"
data_dir3 = "slh_cubes_56_overbound"
train_sets = ["subset0", "subset1", "subset2", "subset3", "subset4", "subset5", "subset6", "subset7", "subset8"]
train_sets2 = ["train", "val"]
#train_sets = ["subset9"]

if "pfilelist_path" in dir() and os.path.exists(pfilelist_path):
	print("read pfilelist from: %s" %(pfilelist_path))
	pfiles = bt.filelist_load(pfilelist_path)
	#pfilelist_file = open(pfilelist_path, "r")
	#pfiles = pfilelist_file.readlines()
	#for pfi in range(len(pfiles)):
	#	pfiles[pfi] = pfiles[pfi][:-1]
	#pfilelist_file.close()
else:
	pfiles = []
	for set in train_sets:
		train_dir = os.path.join(data_dir1, set)
		pdir = os.path.join(train_dir,"npy","*.npy")
		pfiles.extend(glob.glob(pdir))
		#ndir = os.path.join(train_dir,"npy_non","*.npy")
		#nfiles.extend(glob.glob(ndir))
	if "data_dir2" in dir():
		for set in train_sets2:
			train_dir2 = os.path.join(data_dir2, set)
			pdir = os.path.join(train_dir2,"npy","*.npy")
			pfiles.extend(glob.glob(pdir))
	if "data_dir3" in dir():
		pfiles.extend(glob.glob(os.path.join(data_dir3,"npy","*.npy")))
	random.shuffle(pfiles)
bt.filelist_store(pfiles, net_store_path + "/pfilelist.log")
#pfile_storage = open(net_store_path + "/pfilelist.log", "w")
#for pfile in pfiles:
#	pfile_storage.write("%s\n" %(pfile))
#pfile_storage.close()
if "nfilelist_path" in dir() and os.path.exists(nfilelist_path):
	print("read nfilelist from: %s" % (nfilelist_path))
	nfiles = bt.filelist_load(nfilelist_path)
	#nfilelist_file = open(nfilelist_path, "r")
	#nfiles = nfilelist_file.readlines()
	#for nfi in range(len(nfiles)):
	#	nfiles[nfi] = nfiles[nfi][:-1]
	#nfilelist_file.close()
else:
	nfiles = []
	for set in train_sets:
		train_dir = os.path.join(data_dir1, set)
		ndir = os.path.join(train_dir, "npy_non", "*.npy")
		nfiles.extend(glob.glob(ndir))
	random.shuffle(nfiles)
bt.filelist_store(nfiles, net_store_path + "/nfilelist.log")
#nfile_storage = open(net_store_path + "/nfilelist.log", "w")
#for nfile in nfiles:
#	nfile_storage.write("%s\n" %(nfile))
#nfile_storage.close()

num_positive = len(pfiles)
num_negative = len(nfiles)
#num_positive = 10
#num_negative = 200
aug_proportion = (1+2*int(SCALE_AUGMENTATION)) * (1+26*int(TRANSLATION_AUGMENTATION)*(1+2*int(CENTERING))) * (1+9*int(ROTATION_AUGMENTATION)+3*int(FLIP_AUGMENTATION))
if ALL_NEGATIVE:
	np_proportion = num_negative / float(num_positive)
else:
	np_proportion = aug_proportion

#num_negative = 1000 * num_positive
if num_positive==0:
	print("no positive training file found")
	exit()

positive_val_num = int(num_positive * VALIDATION_RATE)
#positive_val_num = 1
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
#negative_importances = 1000*np.ones(shape=[num_negative], dtype=float)

#net construct
volume_inputs = [tf.placeholder(tf.float32, [None, REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0]]),
		 tf.placeholder(tf.float32, [None, REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1]]),
		 tf.placeholder(tf.float32, [None, REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2]])]
volumes_reshape = [tf.reshape(volume_inputs[0], [-1, REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0], 1]),
		   tf.reshape(volume_inputs[1], [-1, REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1], 1]),
		   tf.reshape(volume_inputs[2], [-1, REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2], 1])]
real_label = tf.placeholder(tf.float32, [None, 2])
#r_bn1, b_bn1, w_conv1, w_conv2, out_conv1, out_bn1, hidden_conv1, hidden_conv2, hidden_conv3, out_fc1, out_fc2, softmax_out = tft.volume_bnnet2_l6_56(volume_reshape)
#bn_params = np.load(net_init_path + "/batch_normalization_statistic.npy")
positive_confidence = num_positive_aug / float(num_positive_aug + num_positive * np_proportion)
net_outs0, variables0, _ = tft.volume_bndo_flbias_l5_20(volumes_reshape[0], False, positive_confidence, dropout_rate=0.0, batch_normalization_statistic=False, bn_params=None)
net_outs1, variables1, _ = tft.volume_bndo_flbias_l5_30(volumes_reshape[1], False, positive_confidence, dropout_rate=0.0, batch_normalization_statistic=False, bn_params=None)
net_outs2, variables2, _ = tft.volume_bndo_flbias_l6_40(volumes_reshape[2], False, positive_confidence, dropout_rate=0.0, batch_normalization_statistic=False, bn_params=None)
features = [net_outs0['flattened_out'], net_outs1['flattened_out'], net_outs2['flattened_out']]
net_out, sm_out, fusion_variables = tft.late_fusion(features, True, positive_confidence)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = net_out, labels = real_label)
modulation = tf.pow(real_label[:,0]+tf.pow(tf.constant(-1, dtype=tf.float32), real_label[:,0])*sm_out[:,0], tf.constant(2, dtype=tf.float32)) * (real_label[:,0]+tf.pow(tf.constant(-1, dtype=tf.float32), real_label[:,0])*tf.constant(positive_confidence, tf.float32))
focal_loss = tf.multiply(modulation, cross_entropy)
batch_loss = tf.reduce_mean(focal_loss)
correct_prediction = tf.equal(tf.argmax(sm_out, 1), tf.argmax(real_label, 1))
batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_loss_summary = tf.summary.scalar('train loss', batch_loss)
val_loss_summary = tf.summary.scalar('val loss', batch_loss)
train_acc_summary = tf.summary.scalar('train acc', batch_accuracy)
val_acc_summary = tf.summary.scalar('val acc', batch_accuracy)
train_merge = tf.summary.merge([train_loss_summary, train_acc_summary])
val_merge = tf.summary.merge([val_loss_summary, val_acc_summary])
'''
trains = []
epochs = []
learning_rate = INITIAL_LEARNING_RATE
for ti in range(0, NUM_EPOCH, DECAY_EPOCH):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	trains.append(optimizer.minimize(batch_loss))
	epochs.append(min(NUM_EPOCH-ti, DECAY_EPOCH))
	learning_rate *= DECAY_LEARNING_RATE
'''
epochs = [10, 0, 0]
learning_rates = [0.01, 0.001, 0.0001]
trains = []
for stage in range(len(learning_rates)):
	optimizer = tf.train.GradientDescentOptimizer(learning_rates[stage])
	train = optimizer.minimize(batch_loss)
	trains.append(train)
'''
optimizer1 = tf.train.GradientDescentOptimizer(0.01)
train1 = optimizer1.minimize(batch_loss)
optimizer2 = tf.train.GradientDescentOptimizer(0.001)
train2 = optimizer2.minimize(batch_loss)
optimizer3 = tf.train.GradientDescentOptimizer(0.0001)
train3 = optimizer3.minimize(batch_loss)
trains = [train1, train2, train3]	#training stages of different learning rates
'''

extract_volumes = [ft.partial(mt.extract_volumes, volume_shape=np.int_([REGION_SIZES[0], REGION_SIZES[0], REGION_SIZES[0]]), centering=CENTERING, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION),
		   ft.partial(mt.extract_volumes, volume_shape=np.int_([REGION_SIZES[1], REGION_SIZES[1], REGION_SIZES[1]]), centering=CENTERING, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION),
		   ft.partial(mt.extract_volumes, volume_shape=np.int_([REGION_SIZES[2], REGION_SIZES[2], REGION_SIZES[2]]), centering=CENTERING, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)]

saver0 = tf.train.Saver(variables0)
saver1 = tf.train.Saver(variables1)
saver2 = tf.train.Saver(variables2)
fusion_saver = tf.train.Saver(fusion_variables, max_to_keep=None)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	if 'net_init_files' not in dir() or net_init_files is None:
		print("net randomly initialized")
		sess.run(tf.global_variables_initializer())
	else:
		print("loading net from files:{}" .format(net_init_files))
		saver0.restore(sess, net_init_files[0])
		saver1.restore(sess, net_init_files[1])
		saver2.restore(sess, net_init_files[2])
		if len(net_init_files)>3:
			fusion_saver.restore(sess, net_init_files[3])
		else:
			sess.run(tf.variables_initializer(bt.dict2list(fusion_variables)))
	tensorboard_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
	best_accuracy = 0.0
	best_epoch = 0
	overall_epoch = 0
	training_messages = []
	for train_stage in range(len(trains)):
		train = trains[train_stage]
		for epoch in range(epochs[train_stage]):
			print('epoch:%d/%d' %(overall_epoch+1, epochs[train_stage]))
			#batch_num = 0
			#bn_stats = np.zeros(shape=np.array(bn_params).shape, dtype=float)

			# the number of negative samples is in default larger than the number of positive samples
			ptd_num = 0
			ptd_loss = 0.0
			ptd_accuracy = 0.0
			ntd_num = 0
			ntd_loss = 0.0
			ntd_accuracy = 0.0
			random.shuffle(positive_train_indices)
			random.shuffle(negative_train_indices)
			#positive_batch_size = 1
			positive_batch_size = (1-int(ALL_NEGATIVE))*(int(BATCH_SIZE/2/aug_proportion)-1) + 1	#if augmentatiion implemented then set batch size to 1 or half of BATCH_SIZE
			#negative_batch_size = int(positive_batch_size * np_proportion)
			for pbi in tqdm(range(0, positive_train_num, positive_batch_size)):
				#if AUGMENTATION:
				#	print("training process:%d/%d" %(pbi, positive_train_num))
				posbatchend = min(pbi+positive_batch_size, positive_train_num)
				for pti in range(pbi, posbatchend):
					data_index = positive_train_indices[pti]
					pfile = tpfiles[data_index]
					#pfile = 'luna_cubes_56_overbound/subset5/npy/1.3.6.1.4.1.14519.5.2.1.6279.6001.112740418331256326754121315800_34_ob_annotations.npy'
					if pfile.split('/')[0].find("luna")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "luna_cubes_56_overbound/luna_annotations.csv")
					elif pfile.split('/')[0].find("tianchi")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "tianchi_cubes_56_overbound/tianchi_annotations.csv")
					else:
						patient_uid = mt.get_volume_informations(pfile)[0]
						nodule_diameter = 0
					positive_data = np.load(pfile)
					if "positive_batches" not in dir():
						#positive_batch = mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)
						positive_batches = [extract_volumes[0](positive_data, nodule_diameter=nodule_diameter), extract_volumes[1](positive_data, nodule_diameter=nodule_diameter), extract_volumes[2](positive_data, nodule_diameter=nodule_diameter)]
					else:
						#positive_batch = np.concatenate((positive_batch, mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)), axis=0)
						positive_batches = [np.concatenate((positive_batches[0], extract_volumes[0](positive_data, nodule_diameter=nodule_diameter)), axis=0),
								    np.concatenate((positive_batches[1], extract_volumes[1](positive_data, nodule_diameter=nodule_diameter)), axis=0),
								    np.concatenate((positive_batches[2], extract_volumes[2](positive_data, nodule_diameter=nodule_diameter)), axis=0)]
					del positive_data
				negative_batches = [[], [], []]
				if posbatchend==positive_train_num and ALL_NEGATIVE:
					negbatchend = negative_train_num
				else:
					negbatchend = min(int((pbi+positive_batch_size)*np_proportion), negative_train_num)
				for nti in range(int(pbi*np_proportion), negbatchend):
					data_index = negative_train_indices[nti]
					nfile = tnfiles[data_index]
					negative_data = np.load(nfile)
					ndcenter = np.int_([negative_data.shape[0]/2, negative_data.shape[1]/2, negative_data.shape[2]/2])
					#negative_datas = [negative_data[ndcenter[0]-region_halfs[0]:ndcenter[0]+REGION_SIZES[0]-region_halfs[0], ndcenter[1]-region_halfs[0]:ndcenter[1]+REGION_SIZES[0]-region_halfs[0], ndcenter[2]-region_halfs[0]:ndcenter[2]+REGION_SIZES[0]-region_halfs[0]],
					#		  negative_data[ndcenter[0]-region_halfs[1]:ndcenter[0]+REGION_SIZES[1]-region_halfs[1], ndcenter[1]-region_halfs[1]:ndcenter[1]+REGION_SIZES[1]-region_halfs[1], ndcenter[2]-region_halfs[1]:ndcenter[2]+REGION_SIZES[1]-region_halfs[1]],
					#		  negative_data[ndcenter[0]-region_halfs[2]:ndcenter[0]+REGION_SIZES[2]-region_halfs[2], ndcenter[1]-region_halfs[2]:ndcenter[1]+REGION_SIZES[2]-region_halfs[2], ndcenter[2]-region_halfs[2]:ndcenter[2]+REGION_SIZES[2]-region_halfs[2]]]
					#negative_datas_reshape = [np.reshape(negative_datas[0], (1, negative_datas[0].shape[0], negative_datas[0].shape[1], negative_datas[0].shape[2])),
					#			  np.reshape(negative_datas[1], (1, negative_datas[1].shape[0], negative_datas[1].shape[1], negative_datas[1].shape[2])),
					#			  np.reshape(negative_datas[2], (1, negative_datas[2].shape[0], negative_datas[2].shape[1], negative_datas[2].shape[2]))]

					negative_batches[0].append(negative_data[ndcenter[0]-region_halfs[0]:ndcenter[0]+REGION_SIZES[0]-region_halfs[0], ndcenter[1]-region_halfs[0]:ndcenter[1]+REGION_SIZES[0]-region_halfs[0], ndcenter[2]-region_halfs[0]:ndcenter[2]+REGION_SIZES[0]-region_halfs[0]])
					negative_batches[1].append(negative_data[ndcenter[0]-region_halfs[1]:ndcenter[0]+REGION_SIZES[1]-region_halfs[1], ndcenter[1]-region_halfs[1]:ndcenter[1]+REGION_SIZES[1]-region_halfs[1], ndcenter[2]-region_halfs[1]:ndcenter[2]+REGION_SIZES[1]-region_halfs[1]])
					negative_batches[2].append(negative_data[ndcenter[0]-region_halfs[2]:ndcenter[0]+REGION_SIZES[2]-region_halfs[2], ndcenter[1]-region_halfs[2]:ndcenter[1]+REGION_SIZES[2]-region_halfs[2], ndcenter[2]-region_halfs[2]:ndcenter[2]+REGION_SIZES[2]-region_halfs[2]])
					#if "negative_batches" not in dir():
					#	negative_batches = negative_datas_reshape
					#else:
					#	negative_batches = [np.concatenate((negative_batches[0], negative_datas_reshape[0]), axis=0),
					#			    np.concatenate((negative_batches[1], negative_datas_reshape[1]), axis=0),
					#			    np.concatenate((negative_batches[2], negative_datas_reshape[2]), axis=0)]
					del negative_data
								    
				if positive_batches[0].shape[0]!=positive_batches[1].shape[0] or positive_batches[1].shape[0]!=positive_batches[2].shape[0]:
					print('batches inconsistency')
					continue
				posbatch_size = positive_batches[0].shape[0]
				allbatch_size = positive_batches[0].shape[0] + len(negative_batches[0])
				train_datas = [np.zeros(shape=(allbatch_size, positive_batches[0].shape[1], positive_batches[0].shape[2], positive_batches[0].shape[3]), dtype=float),
					       np.zeros(shape=(allbatch_size, positive_batches[1].shape[1], positive_batches[1].shape[2], positive_batches[1].shape[3]), dtype=float),
					       np.zeros(shape=(allbatch_size, positive_batches[2].shape[1], positive_batches[2].shape[2], positive_batches[2].shape[3]), dtype=float)]
				train_label = np.zeros(shape=(allbatch_size, 2), dtype=float)
				batch_random = np.random.permutation(allbatch_size)
				#batch_random = np.arange(positive_batch.shape[0]+negative_batch.shape[0])
				for bi in range(batch_random.size):
					batch_index = batch_random[bi]
					if batch_index<posbatch_size:
						train_datas[0][bi] = (positive_batches[0][batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_datas[1][bi] = (positive_batches[1][batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_datas[2][bi] = (positive_batches[2][batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_label[bi][0] = 1
					else:
						train_datas[0][bi] = (negative_batches[0][batch_index-posbatch_size] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_datas[1][bi] = (negative_batches[1][batch_index-posbatch_size] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_datas[2][bi] = (negative_batches[2][batch_index-posbatch_size] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						train_label[bi][1] = 1
				for bi in range(0, allbatch_size, BATCH_SIZE):
					batch_size = min(BATCH_SIZE, allbatch_size-bi)
					data_batches = [train_datas[0][bi:bi+batch_size], train_datas[1][bi:bi+batch_size], train_datas[2][bi:bi+batch_size]]
					label_batch = train_label[bi:bi+batch_size]
					_, losses, accuracies, train_summary = sess.run([train, focal_loss, correct_prediction, train_merge], {volume_inputs[0]: data_batches[0], volume_inputs[1]: data_batches[1], volume_inputs[2]: data_batches[2], real_label: label_batch})
					#parameters2 = sess.run(variables, {volume_input: data_batch, real_label: label_batch})
					tensorboard_writer.add_summary(train_summary, epoch)
					for di in range(batch_size):
						if label_batch[di][0]>label_batch[di][1]:
							ptd_num += 1
							ptd_loss += losses[di]
							ptd_accuracy += accuracies[di]
						else:
							ntd_num += 1
							ntd_loss += losses[di]
							ntd_accuracy += accuracies[di]
					#if batch_size > 1:
					#	batch_num += 1
					#	bn_stat = np.array(bn_stat)
					#	bn_stat[:,1] = bn_stat[:,1] * batch_size / (batch_size-1)	#unbiased variance estimate
					#	bn_stats += bn_stat

				if 'negative_batches' in dir():
					del negative_batches
				del positive_batches
				del train_datas

			if ptd_num > 0:
				ptd_loss /= ptd_num
				ptd_accuracy /= ptd_num
			if ntd_num > 0:
				ntd_loss /= ntd_num
				ntd_accuracy /= ntd_num

			pvd_num = 0
			pvd_loss = 0.0
			pvd_accuracy = 0.0
			nvd_num = 0
			nvd_loss = 0.0
			nvd_accuracy = 0.0
			random.shuffle(positive_val_indices)
			random.shuffle(negative_val_indices)
			#positive_batch_size = 1
			positive_batch_size = (1-int(AUGMENTATION or ALL_NEGATIVE))*(int(BATCH_SIZE/2)-1) + 1	#if augmentatiion implemented then set batch size to 1 or half of BATCH_SIZE
			#negative_batch_size = int(positive_batch_size * np_proportion)
			for pbi in tqdm(range(0, positive_val_num, positive_batch_size)):
				posbatchend = min(pbi+positive_batch_size, positive_val_num)
				for pvi in range(pbi, posbatchend):
					data_index = positive_val_indices[pvi]
					pfile = vpfiles[data_index]
					#pfile = 'luna_cubes_56_overbound/subset5/npy/1.3.6.1.4.1.14519.5.2.1.6279.6001.112740418331256326754121315800_34_ob_annotations.npy'
					if pfile.split('/')[0].find("luna")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "luna_cubes_56_overbound/luna_annotations.csv")
					elif pfile.split('/')[0].find("tianchi")>=0:
						patient_uid, nodule_diameter = mt.get_annotation_informations(pfile, "tianchi_cubes_56_overbound/tianchi_annotations.csv")
					else:
						patient_uid = mt.get_volume_informations(pfile)[0]
						nodule_diameter = 0
					positive_data = np.load(pfile)
					if "positive_batches" not in dir():
						#positive_batch = mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)
						positive_batches = [extract_volumes[0](positive_data, nodule_diameter=nodule_diameter), extract_volumes[1](positive_data, nodule_diameter=nodule_diameter), extract_volumes[2](positive_data, nodule_diameter=nodule_diameter)]
					else:
						#positive_batch = np.concatenate((positive_batch, mt.extract_volumes(positive_data, centering=CENTERING, nodule_diameter=nodule_diameter, scale_augment=SCALE_AUGMENTATION, translation_augment=TRANSLATION_AUGMENTATION, rotation_augment=ROTATION_AUGMENTATION, flip_augment=FLIP_AUGMENTATION)), axis=0)
						positive_batches = [np.concatenate((positive_batches[0], extract_volumes[0](positive_data, nodule_diameter=nodule_diameter)), axis=0),
								    np.concatenate((positive_batches[1], extract_volumes[1](positive_data, nodule_diameter=nodule_diameter)), axis=0),
								    np.concatenate((positive_batches[2], extract_volumes[2](positive_data, nodule_diameter=nodule_diameter)), axis=0)]
					del positive_data
				negative_batches = [[], [], []]
				if posbatchend==positive_val_num and ALL_NEGATIVE:
					negbatchend = negative_val_num
				else:
					negbatchend = min(int((pbi+positive_batch_size)*np_proportion), negative_val_num)
				for nvi in range(int(pbi*np_proportion), negbatchend):
					data_index = negative_val_indices[nvi]
					nfile = vnfiles[data_index]
					negative_data = np.load(nfile)
					ndcenter = np.int_([negative_data.shape[0]/2, negative_data.shape[1]/2, negative_data.shape[2]/2])
					#negative_datas = [negative_data[ndcenter[0]-region_halfs[0]:ndcenter[0]+REGION_SIZES[0]-region_halfs[0], ndcenter[1]-region_halfs[0]:ndcenter[1]+REGION_SIZES[0]-region_halfs[0], ndcenter[2]-region_halfs[0]:ndcenter[2]+REGION_SIZES[0]-region_halfs[0]],
					#		  negative_data[ndcenter[0]-region_halfs[1]:ndcenter[0]+REGION_SIZES[1]-region_halfs[1], ndcenter[1]-region_halfs[1]:ndcenter[1]+REGION_SIZES[1]-region_halfs[1], ndcenter[2]-region_halfs[1]:ndcenter[2]+REGION_SIZES[1]-region_halfs[1]],
					#		  negative_data[ndcenter[0]-region_halfs[2]:ndcenter[0]+REGION_SIZES[2]-region_halfs[2], ndcenter[1]-region_halfs[2]:ndcenter[1]+REGION_SIZES[2]-region_halfs[2], ndcenter[2]-region_halfs[2]:ndcenter[2]+REGION_SIZES[2]-region_halfs[2]]]
					#negative_datas_reshape = [np.reshape(negative_datas[0], (1, negative_datas[0].shape[0], negative_datas[0].shape[1], negative_datas[0].shape[2])),
					#			  np.reshape(negative_datas[1], (1, negative_datas[1].shape[0], negative_datas[1].shape[1], negative_datas[1].shape[2])),
					#			  np.reshape(negative_datas[2], (1, negative_datas[2].shape[0], negative_datas[2].shape[1], negative_datas[2].shape[2]))]

					negative_batches[0].append(negative_data[ndcenter[0]-region_halfs[0]:ndcenter[0]+REGION_SIZES[0]-region_halfs[0], ndcenter[1]-region_halfs[0]:ndcenter[1]+REGION_SIZES[0]-region_halfs[0], ndcenter[2]-region_halfs[0]:ndcenter[2]+REGION_SIZES[0]-region_halfs[0]])
					negative_batches[1].append(negative_data[ndcenter[0]-region_halfs[1]:ndcenter[0]+REGION_SIZES[1]-region_halfs[1], ndcenter[1]-region_halfs[1]:ndcenter[1]+REGION_SIZES[1]-region_halfs[1], ndcenter[2]-region_halfs[1]:ndcenter[2]+REGION_SIZES[1]-region_halfs[1]])
					negative_batches[2].append(negative_data[ndcenter[0]-region_halfs[2]:ndcenter[0]+REGION_SIZES[2]-region_halfs[2], ndcenter[1]-region_halfs[2]:ndcenter[1]+REGION_SIZES[2]-region_halfs[2], ndcenter[2]-region_halfs[2]:ndcenter[2]+REGION_SIZES[2]-region_halfs[2]])
					#if "negative_batches" not in dir():
					#	negative_batches = negative_datas_reshape
					#else:
					#	negative_batches = [np.concatenate((negative_batches[0], negative_datas_reshape[0]), axis=0),
					#			    np.concatenate((negative_batches[1], negative_datas_reshape[1]), axis=0),
					#			    np.concatenate((negative_batches[2], negative_datas_reshape[2]), axis=0)]
					del negative_data
								    
				if positive_batches[0].shape[0]!=positive_batches[1].shape[0] or positive_batches[1].shape[0]!=positive_batches[2].shape[0]:
					print('batches inconsistency')
					continue
				posbatch_size = positive_batches[0].shape[0]
				allbatch_size = positive_batches[0].shape[0] + len(negative_batches[0])
				val_datas = [np.zeros(shape=(allbatch_size, positive_batches[0].shape[1], positive_batches[0].shape[2], positive_batches[0].shape[3]), dtype=float),
					       np.zeros(shape=(allbatch_size, positive_batches[1].shape[1], positive_batches[1].shape[2], positive_batches[1].shape[3]), dtype=float),
					       np.zeros(shape=(allbatch_size, positive_batches[2].shape[1], positive_batches[2].shape[2], positive_batches[2].shape[3]), dtype=float)]
				val_label = np.zeros(shape=(allbatch_size, 2), dtype=float)
				batch_random = np.random.permutation(allbatch_size)
				#batch_random = np.arange(positive_batch.shape[0]+negative_batch.shape[0])
				for bi in range(batch_random.size):
					batch_index = batch_random[bi]
					if batch_index<posbatch_size:
						val_datas[0][bi] = (positive_batches[0][batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_datas[1][bi] = (positive_batches[1][batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_datas[2][bi] = (positive_batches[2][batch_index] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_label[bi][0] = 1
					else:
						val_datas[0][bi] = (negative_batches[0][batch_index-posbatch_size] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_datas[1][bi] = (negative_batches[1][batch_index-posbatch_size] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_datas[2][bi] = (negative_batches[2][batch_index-posbatch_size] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) - PIXEL_MEAN
						val_label[bi][1] = 1
				for bi in range(0, allbatch_size, BATCH_SIZE):
					batch_size = min(BATCH_SIZE, allbatch_size-bi)
					data_batches = [val_datas[0][bi:bi+batch_size], val_datas[1][bi:bi+batch_size], val_datas[2][bi:bi+batch_size]]
					label_batch = val_label[bi:bi+batch_size]
					losses, accuracies, val_summary = sess.run([focal_loss, correct_prediction, val_merge], {volume_inputs[0]: data_batches[0], volume_inputs[1]: data_batches[1], volume_inputs[2]: data_batches[2], real_label: label_batch})
					#parameters2 = sess.run(variables, {volume_input: data_batch, real_label: label_batch})
					tensorboard_writer.add_summary(val_summary, epoch)
					for di in range(batch_size):
						if label_batch[di][0]>label_batch[di][1]:
							pvd_num += 1
							pvd_loss += losses[di]
							pvd_accuracy += accuracies[di]
						else:
							nvd_num += 1
							nvd_loss += losses[di]
							nvd_accuracy += accuracies[di]
					#if batch_size > 1:
					#	batch_num += 1
					#	bn_stat = np.array(bn_stat)
					#	bn_stat[:,1] = bn_stat[:,1] * batch_size / (batch_size-1)	#unbiased variance estimate
					#	bn_stats += bn_stat

				if 'negative_batches' in dir():
					del negative_batches
				del positive_batches
				del val_datas

			if pvd_num > 0:
				pvd_loss /= pvd_num
				pvd_accuracy /= pvd_num
			if nvd_num > 0:
				nvd_loss /= nvd_num
				nvd_accuracy /= nvd_num

			#tensorboard_writer.add_summary(summary, epoch)
			print("positive_train_loss:%f positive_train_acc:%f negative_train_loss:%f negative_train_acc:%f\npositive_val_loss:%f positive_val_acc:%f negative_val_loss:%f negative_val_acc:%f" % (
				ptd_loss, ptd_accuracy, ntd_loss,
				ntd_accuracy, pvd_loss, pvd_accuracy, nvd_loss,
				nvd_accuracy))
			if (pvd_accuracy+nvd_accuracy)/2>=best_accuracy:
				best_accuracy = (pvd_accuracy+nvd_accuracy)/2
				best_epoch = overall_epoch
				#best_dir = net_store_path+"/"+net_store_name+"_best"
				best_dir = net_store_path+"/"+"best"
				if os.access(best_dir, os.F_OK):
					shutil.rmtree(best_dir)
				os.mkdir(best_dir)
				#saver.save(sess, best_dir+"/"+net_store_name+"_best")
				fusion_saver.save(sess, best_dir+"/"+"best")
			training_messages.append(
				[overall_epoch + 1, ptd_loss, ptd_accuracy, ntd_loss,
				 ntd_accuracy, pvd_loss, pvd_accuracy, nvd_loss,
				 nvd_accuracy])
			# write training process to a logger file
			logger = open(net_store_path + "/" + net_store_name + ".log", 'w')
			logger.write("random validation with overall negative samples\n")
			logger.write("validation rate:{}\n" .format(VALIDATION_RATE))
			logger.write("batch size:{}\n" .format(BATCH_SIZE))
			logger.write("region sizes:{}\n" .format(REGION_SIZES))
			logger.write("max bound:{}\n" .format(MAX_BOUND))
			logger.write("min bound:{}\n" .format(MIN_BOUND))
			logger.write("pixel mean:{}\n" .format(PIXEL_MEAN))
			logger.write("scale augmentation:{}\n" .format(SCALE_AUGMENTATION))
			logger.write("translation augmentation:{}\n" .format(TRANSLATION_AUGMENTATION))
			logger.write("rotation augmentation:{}\n" .format(ROTATION_AUGMENTATION))
			logger.write("flip augmentation:{}\n" .format(FLIP_AUGMENTATION))
			logger.write("centering augmentation:{}\n" .format(CENTERING))
			# logger.write("initial_learning_rate:%f decay_rate:%f decay_epoch:%d\n" %(INITIAL_LEARNING_RATE, DECAY_LEARNING_RATE, DECAY_EPOCH))
			logger.write(
				"learning_rates:%f %f %f\n" % (learning_rates[0], learning_rates[1], learning_rates[2]))
			logger.write("epochs:%d %d %d\n" % (epochs[0], epochs[1], epochs[2]))
			logger.write(
				"epoch pos_train_loss pos_train_acc neg_train_loss neg_train_acc pos_val_loss pos_val_acc neg_val_loss neg_val_acc\n")
			for tm in range(len(training_messages)):
				logger.write("%d %f %f %f %f %f %f %f %f\n" % (
				training_messages[tm][0], training_messages[tm][1], training_messages[tm][2],
				training_messages[tm][3], training_messages[tm][4], training_messages[tm][5],
				training_messages[tm][6], training_messages[tm][7], training_messages[tm][8]))
			logger.write("best epoch:%d" %(best_epoch+1))
			logger.close()
			if SNAPSHOT_EPOCH>0 and (overall_epoch+1)%SNAPSHOT_EPOCH==0:
				#snapshot_dir = net_store_path+"/"+net_store_name+"_epoch"+str(overall_epoch+1)
				snapshot_dir = net_store_path+"/"+"epoch"+str(overall_epoch+1)
				if os.access(snapshot_dir, os.F_OK):
					shutil.rmtree(snapshot_dir)
				os.mkdir(snapshot_dir)
				fusion_saver.save(sess, snapshot_dir+"/"+"epoch"+str(overall_epoch+1))
				#np.save(snapshot_dir+"/batch_normalization_statistic.npy", bn_stats)
			overall_epoch += 1
	fusion_saver.save(sess, net_store_path + "/" + net_store_name)
	#np.save(net_store_path + "/batch_normalization_statistic.npy", bn_stats)
sess.close()
print("Overall training done!")
print("The network is saved as:%s" %(net_store_name))
