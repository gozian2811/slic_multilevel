import os
import math
import random
import copy
import array
import dicom
import scipy.ndimage
import pandas as pd
import numpy as np
import SimpleITK as sitk

def get_serie_uid(filepath):
	filename = os.path.basename(filepath)
	fileparts = os.path.splitext(filename)
	return fileparts[0]

def get_volume_informations(filepath):
	filename = os.path.basename(filepath)
	fileparts = os.path.splitext(filename)
	fileinfos = fileparts[0].split("_")
	return fileinfos

def get_annotations(patient_uid, annofile):
	annotations = pd.read_csv(annofile)
	annolines = (annotations["seriesuid"].values==patient_uid).nonzero()[0]
	annolist = []
	for annoline in annolines:
		coordX = annotations["coordX"].values[annoline]
		coordY = annotations["coordY"].values[annoline]
		coordZ = annotations["coordZ"].values[annoline]
		diameter_mm = annotations["diameter_mm"].values[annoline]
		annolist.append([coordX, coordY, coordZ, diameter_mm])
	return annolist

def get_annotation_informations(filepath, annofile):
	fileinfos = get_volume_informations(filepath)
	patient_uid = fileinfos[0]
	annoline = int(fileinfos[1])

	annotations = pd.read_csv(annofile)
	anno_uid = annotations["seriesuid"].values[annoline]
	if patient_uid != anno_uid:
		return "", 0
	nodule_diameter = float(annotations["diameter_mm"].values[annoline])
	return anno_uid, nodule_diameter

def read_dicom_scan(case_path):
	reader = sitk.ImageSeriesReader()
	dicom_infos = []
	serie_count = {}
	max_serie = -1
	max_serie_count = 0
	for s in os.listdir(case_path):
		filename = case_path + '/' + s
		info = dicom.read_file(filename)
		siuid = info.SeriesInstanceUID
		#if filename=='./SLH_data/20170609/P05471399/0021bfa7.dcm':
		#	print("%s" %(siuid))
		instancenumber = info.InstanceNumber
		dicom_infos.append([filename, siuid, instancenumber])
		if siuid in serie_count:
			serie_count[siuid] += 1
		else:
			serie_count[siuid] = 1
		if max_serie_count < serie_count[siuid]:
			max_serie_count = serie_count[siuid]
			max_serie = siuid
	dicom_series = []
	instance_numbers = []
	for filename, siuid, instancenumber in dicom_infos:
		if siuid == max_serie:
			if len(dicom_series)==0:
				dicom_series.append(filename)
				instance_numbers.append(instancenumber)
			else:
				for ds in range(len(dicom_series)):
					if instancenumber<instance_numbers[ds]:
						dicom_series.insert(ds, filename)
						instance_numbers.insert(ds, instancenumber)
						break
				if ds>=len(dicom_series)-1:
					dicom_series.append(filename)
					instance_numbers.append(instancenumber)

	reader.SetFileNames(dicom_series)
	full_image_info = reader.Execute()
	full_scan = sitk.GetArrayFromImage(full_image_info)
	return full_scan, full_image_info, info.PatientID

def write_mhd_file(mhdfile, data, dsize):
	def write_meta_header(filename, meta_dict):
		header = ''
		# do not use tags = meta_dict.keys() because the order of tags matters
		tags = ['ObjectType', 'NDims', 'BinaryData',
			'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
			'TransformMatrix', 'Offset', 'CenterOfRotation',
			'AnatomicalOrientation',
			'ElementSpacing',
			'DimSize',
			'ElementType',
			'ElementDataFile',
			'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
		for tag in tags:
			if tag in meta_dict.keys():
				header += '%s = %s\n' % (tag, meta_dict[tag])
		f = open(filename, 'w')
		f.write(header)
		f.close()

	def dump_raw_data(filename, data):
		""" Write the data into a raw format file. Big endian is always used. """
		# Begin 3D fix
		data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
		# End 3D fix
		rawfile = open(filename, 'wb')
		a = array.array('f')
		for o in data:
			a.fromlist(list(o))
		# if is_little_endian():
		#    a.byteswap()
		a.tofile(rawfile)
		rawfile.close()
	assert (mhdfile[-4:] == '.mhd')
	meta_dict = {}
	meta_dict['ObjectType'] = 'Image'
	meta_dict['BinaryData'] = 'True'
	meta_dict['BinaryDataByteOrderMSB'] = 'False'
	meta_dict['ElementType'] = 'MET_FLOAT'
	meta_dict['NDims'] = str(len(dsize))
	meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
	meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd', '.raw')
	write_meta_header(mhdfile, meta_dict)
	pwd = os.path.split(mhdfile)[0]
	if pwd:
		data_file = pwd + '/' + meta_dict['ElementDataFile']
	else:
		data_file = meta_dict['ElementDataFile']
	dump_raw_data(data_file, data)

def medical_normalization(x, max_bound = 512.0, min_bound = -1024.0, pixel_mean = 0.25, crop = False, input_copy = True):
	if input_copy:
		x = copy.copy(x)
	x = (x - min_bound) / float(max_bound - min_bound)
	if crop:
		x[x>1] = 1
		x[x<0] = 0
	x = x - pixel_mean
	return x
	
'''
def coord_overflow(coord, shape):
	upbound = shape - coord	
	if coord[coord<0].size>0 or upbound[upbound<=0].size>0:
		return True
	else:
		return False
'''
def coord_overflow(coord, shape):
	for i in range(len(coord)):
		if coord[i]<0 or coord[i]>=shape[i]:
			return True
	return False

def resample(image, old_spacing, new_spacing=np.array([1, 1, 1])):
	resize_factor = old_spacing / new_spacing
	new_real_shape = image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = old_spacing / real_resize_factor
	if image.shape[0]<1000:
        	image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
	else:
		num_batch = int(math.ceil(image.shape[0]/1000))
		for b in range(num_batch):
			image_batch = image[b*1000:min((b+1)*1000,image.shape[0]), :, :]
			image_batch = scipy.ndimage.interpolation.zoom(image_batch, real_resize_factor, mode='nearest')
			if 'new_image' in dir():
				new_image = np.append(new_image, image_batch, axis=0)
			else:
				new_image = image_batch
		image = new_image

	return image, new_spacing

def region_valid(voxels, threshold_ratio=0.99):
	num_voxel = voxels.size
	num_tissue = voxels[voxels>-600].size
	if num_tissue < num_voxel*threshold_ratio:
		return True
	else:
		#too much tissue in this region, there's little possibility for nodules to exist
		return False

def extract_volumes(volume_overbound, volume_shape, nodule_diameter=0, centering=True, scale_augment=False, translation_augment=False, rotation_augment=False, flip_augment=False):
	#volume_shape = np.int_(np.array(volume_overbound.shape)/2)
	v_center = np.int_(np.array(volume_overbound.shape)/2)
	if scale_augment:
		#the scale indicates the real size of the cropped box
		if nodule_diameter>44:
			scales = [1.0, 1.25]
		elif nodule_diameter<10 and nodule_diameter>0:
			scales = [0.8, 1.0]
		else:
			scales = [0.8,1.0,1.25]
	else:
		scales = [1.0]
	if translation_augment and nodule_diameter>0:
		translations = np.array([[0,0,0],[0,0,1],[0,0,-1], [0,1,0],[0,math.sqrt(0.5),math.sqrt(0.5)],[0,math.sqrt(0.5),-math.sqrt(0.5)], [0,-1,0],[0,-math.sqrt(0.5),math.sqrt(0.5)],[0,-math.sqrt(0.5),-math.sqrt(0.5)],
				         [1,0,0],[math.sqrt(0.5),0,math.sqrt(0.5)],[math.sqrt(0.5),0,-math.sqrt(0.5)], [math.sqrt(0.5),math.sqrt(0.5),0],[math.sqrt(0.3333),math.sqrt(0.3333),math.sqrt(0.3333)],[math.sqrt(0.3333),math.sqrt(0.3333),-math.sqrt(0.3333)], [math.sqrt(0.5),-math.sqrt(0.5),0],[math.sqrt(0.3333),-math.sqrt(0.3333),math.sqrt(0.3333)],[math.sqrt(0.3333),-math.sqrt(0.3333),-math.sqrt(0.3333)],
				         [-1,0,0],[-math.sqrt(0.5),0,math.sqrt(0.5)],[-math.sqrt(0.5),0,-math.sqrt(0.5)], [-math.sqrt(0.5),math.sqrt(0.5),0],[-math.sqrt(0.3333),math.sqrt(0.3333),math.sqrt(0.3333)],[-math.sqrt(0.3333),math.sqrt(0.3333),-math.sqrt(0.3333)], [-math.sqrt(0.5),-math.sqrt(0.5),0],[-math.sqrt(0.3333),-math.sqrt(0.3333),math.sqrt(0.3333)],[-math.sqrt(0.3333),-math.sqrt(0.3333),-math.sqrt(0.3333)]])
		#translations = np.array([[0,0,0],[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]])
		num_translations = 27
		rt = np.zeros(num_translations, dtype=int)
		rt[1:num_translations] = np.random.choice(range(1,len(translations)), num_translations-1, False)
		rt = np.sort(rt)
	else:
		translations = np.array([[0,0,0]])
		rt = np.array([0])

	#num_translations = min(5, len(translations))
	for s in range(len(scales)):
		#rt = np.zeros(num_translations, dtype=int)
		#rt[1:num_translations] = np.random.choice(range(1,len(translations)), num_translations-1, False)
		#rt = np.sort(rt)
		for t in range(rt.size):
			scale = scales[s]
			box_size = np.int_(np.ceil(volume_shape*scale))
			window_size = np.array(box_size/2, dtype=int)
			if nodule_diameter>0 and t!=0:
				if centering:
					transscales = np.array([nodule_diameter * 0.3])
				else:
					tsnum = math.sqrt(box_size.max()/nodule_diameter)
					step = box_size.max() / 2 / tsnum
					transscales = np.arange(1.0, tsnum) * step
			else:
				transscales = np.array([0])
			for ts in range(len(transscales)):
				transscale = transscales[ts]
				translation = np.array(transscale*translations[rt[t]], dtype=int)	#the translation step cooperating with the nodule_diameter to ensure the translation being within the range of the nodule boundary
				tnz = (np.absolute(translation)>1).nonzero()[0]
				if tnz.size==0 and t!=0:	#the translation is too tiny to distinguish
					#print('diameter:{} scale:{} translation:{} the translation is invisible' .format(nodule_diameter, scale, translation))
					continue
				tob = ((box_size/2-translation)>nodule_diameter/2).nonzero()[0]
				if not centering and tob.size==0:
					#print('diameter:{} scale:{} translation:{} nodule out of box' .format(nodule_diameter, scale, translation))
					continue

				zyx_1 = v_center + translation - window_size  #the order of indices is [Z, Y, X]
				zyx_2 = v_center + translation + box_size - window_size
				if coord_overflow(zyx_1, volume_overbound.shape) or coord_overflow(zyx_2, volume_overbound.shape):
					#print('diameter:{} scale:{} translation:{} the region is out of the bound of the volume' .format(nodule_diameter, scale, translation))
					continue
				nodule_box = np.zeros(shape=volume_shape, dtype=int)  # ---nodule_box_size = 45
				img_crop = volume_overbound[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
				img_crop[img_crop<-1024] = -1024  # the voxel value below -1024 is set to -1024
				if scale==1.0:
					img_crop_rescaled = img_crop
				else:
					img_crop_rescaled, rescaled_spacing = resample(img_crop, np.array([1,1,1]), np.array([scale,scale,scale]))
				padding_shape = np.array((img_crop_rescaled.shape-volume_shape)/2, dtype=int)
				nodule_box = img_crop_rescaled[padding_shape[0]:padding_shape[0]+volume_shape[0], padding_shape[1]:padding_shape[1]+volume_shape[1], padding_shape[2]:padding_shape[2]+volume_shape[2]]
				if 'volume_batch' not in dir():
					volume_batch = nodule_box.reshape((1, volume_shape[0], volume_shape[1], volume_shape[2]))
				else:
					volume_batch = np.concatenate((volume_batch, nodule_box.reshape((1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
				if rotation_augment:
					rot_box = np.rot90(nodule_box, k=1, axes=(2, 1))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=2, axes=(2, 1))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=3, axes=(2, 1))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=1, axes=(2, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=2, axes=(2, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=3, axes=(2, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=1, axes=(1, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=2, axes=(1, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					rot_box = np.rot90(nodule_box, k=3, axes=(1, 0))
					volume_batch = np.concatenate((volume_batch, rot_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
				if flip_augment:
					flip_box = nodule_box[::-1,:,:]
					volume_batch = np.concatenate((volume_batch, flip_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					flip_box = nodule_box[:,::-1,:]
					volume_batch = np.concatenate((volume_batch, flip_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					flip_box = nodule_box[:,:,::-1]
					volume_batch = np.concatenate((volume_batch, flip_box.reshape(
						(1, volume_shape[0], volume_shape[1], volume_shape[2]))), axis=0)
					
	if 'volume_batch' not in dir():
		print('volume extraction failed')
		np.save('error.npy', volume_overbound)
	return volume_batch

def make_patchs(voxels):
	width, length, height = voxels.shape
	patch_size = np.min(voxels.shape)
	patchs = np.zeros(shape=(9,patch_size,patch_size), dtype = float)
	patchs[0] = voxels[:,:,int(height/2)]
	patchs[1] = voxels[:,int(length/2),:]
	patchs[2] = voxels[int(width/2),:,:]
	for h in range(height):
		patchs[3,:,h] = voxels[:,h,h]
		patchs[4,h,:] = voxels[h,:,h]
		patchs[5,:,h] = voxels[:,h,height-h-1]
		patchs[6,h,:] = voxels[h,:,height-h-1]
	for w in range(width):
		patchs[7,w,:] = voxels[w,w,:]
		patchs[8,w,:] = voxels[width-w-1,w,:]
	
	return patchs

def concatenate_patchs(patchs, num_channels=3):
	img_height = patchs.shape[1]
	img_width = patchs.shape[2]
	half_width = int(img_width/2)
	half_height = int(img_height/2)

	#3 orders of patch indices
	patch_indices_list = []
	patch_indices = [ind for ind in range(9)]
	patch_indices_list.append(patch_indices)
	patch_indices_list.append(copy.copy(patch_indices))
	patch_indices_list.append(copy.copy(patch_indices))
	#random.shuffle(patch_indices_list[0])
	random.shuffle(patch_indices_list[1])
	patch_indices_list[2].reverse()

	patch_chan = np.zeros(shape=(4*img_height,4*img_width,num_channels), dtype=float)
	for c in range(num_channels):
		patch_indices = patch_indices_list[c]
		aug_patch = np.ndarray(shape=(9*img_height, 9*img_width), dtype=float)
		for h in range(3):
			for w in range(3):
				for hi in range(3):
					for wi in range(3):
						aug_patch[(h*3+hi)*img_height:(h*3+hi+1)*img_height, (w*3+wi)*img_width:(w*3+wi+1)*img_width] = patchs[patch_indices[hi*3+wi]]
		patch_chan[:,:,c] = aug_patch[2*img_height+half_height:7*img_height-half_height, 2*img_width+half_width:7*img_width-half_width]

	return patch_chan

def nodule_cluster(nodule_centers, scale, iterate=False):
	print("Clustering:")
	clusters=[]
	l=len(nodule_centers)
	if l==0:
		return clusters
	center_index_cluster = 0 - np.ones(len(nodule_centers), dtype=int)
	#initial clustering
	point = nodule_centers[l-1]	#point is a list
	center_index_cluster[l-1] = 0
	clusters.append([point, point, 1])
	for i in range(l-1):
		point = nodule_centers[i] #The current point to be clustered
		flag = 0
		nearsqdist = scale * scale
		nearcand = -1
		#find the older cluster
		for j in range(len(clusters)):
			#calculate the distance with only coordination but prediction
			sqdist = (point[0]-clusters[j][0][0])*(point[0]-clusters[j][0][0]) + (point[1]-clusters[j][0][1])*(point[1]-clusters[j][0][1]) + (point[2]-clusters[j][0][2])*(point[2]-clusters[j][0][2])
			if sqdist<scale*scale and sqdist<nearsqdist: #which means we should add the point into this cluster
				#Notice the type that cluster is a list so we need to write a demo
				nearsqdist = sqdist
				nearcand = j
				flag=1
		if flag==1:
			clusters[nearcand][1] = [(clusters[nearcand][1][0]+point[0]),
                				(clusters[nearcand][1][1]+point[1]),
                				(clusters[nearcand][1][2]+point[2]),
						(clusters[nearcand][1][3]+point[3])]
			clusters[nearcand][2] = clusters[nearcand][2]+1
			clusters[nearcand][0] = [(clusters[nearcand][1][0])/clusters[nearcand][2],
                				(clusters[nearcand][1][1])/clusters[nearcand][2],
                				(clusters[nearcand][1][2])/clusters[nearcand][2],
						(clusters[nearcand][1][3])/clusters[nearcand][2]]
			center_index_cluster[i] = nearcand
		else:
			# create a new cluster
			center_index_cluster[i] = len(clusters)
			clusters.append([point, point, 1])
	
	if iterate:
		#rearrange the clusters by iterations
		converge = False
		while not converge:
			converge = True
			for i in range(l):
				point = nodule_centers[i] #The current point to be clustered
				nearsqdist = scale*scale
				nearcand = -1
				#find the older cluster
				for j in range(len(clusters)):
					if clusters[j][2]<=0:
		    				continue
					#calculate the distance with only coordination but prediction
					sqdist = (point[0]-clusters[j][0][0])*(point[0]-clusters[j][0][0]) + (point[1]-clusters[j][0][1])*(point[1]-clusters[j][0][1]) + (point[2]-clusters[j][0][2])*(point[2]-clusters[j][0][2])
					if sqdist<nearsqdist: #which means we should add the point into this cluster
						#Notice the type that cluster is a list so we need to write a demo
						nearsqdist = sqdist
						nearcand = j
				if nearcand>=0 and nearcand!=center_index_cluster[i]:
					converge = False
					oldcand = center_index_cluster[i]
					if oldcand>=0:
						clusters[oldcand][1] = [(clusters[oldcand][1][0] - point[0]),
												 (clusters[oldcand][1][1] - point[1]),
												 (clusters[oldcand][1][2] - point[2]),
												 (clusters[oldcand][1][3] - point[3])]
						clusters[oldcand][2] = clusters[oldcand][2] - 1
						clusters[oldcand][0] = [(clusters[oldcand][1][0]) / clusters[oldcand][2],
												(clusters[oldcand][1][1]) / clusters[oldcand][2],
												(clusters[oldcand][1][2]) / clusters[oldcand][2],
												(clusters[oldcand][1][3]) / clusters[oldcand][2]]
					clusters[nearcand][1] = [(clusters[nearcand][1][0]+point[0]),
								(clusters[nearcand][1][1]+point[1]),
								(clusters[nearcand][1][2]+point[2]),
								(clusters[nearcand][1][3]+point[3])]
					clusters[nearcand][2] = clusters[nearcand][2]+1
					clusters[nearcand][0] = [(clusters[nearcand][1][0]) / clusters[nearcand][2],
											 (clusters[nearcand][1][1]) / clusters[nearcand][2],
											 (clusters[nearcand][1][2]) / clusters[nearcand][2],
											 (clusters[nearcand][1][3]) / clusters[nearcand][2]]
					center_index_cluster[i] = nearcand
	solid_clusters = [c for c in clusters if c[2]>0]
	print('Clustering Done')

	return solid_clusters
