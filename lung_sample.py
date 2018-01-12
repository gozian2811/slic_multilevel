#!/usr/bin/env python
# encoding: utf-8


"""
@version: python 2.7
@author: Sober.JChen
@license: Apache Licence 
@contact: jzcjedu@foxmail.com
@software: PyCharm
@file: crop_save_and_view_nodules_in_3d.py
@time: 2017/3/14 13:15
"""

# ToDo ---这个脚本运行时请先根据预定义建好文件夹，并将candidates.csv文件的class头改成nodule_class并存为candidates_class.csv，否则会报错。

# ======================================================================
# Program:   Diffusion Weighted MRI Reconstruction
# Link:      https://code.google.com/archive/p/diffusion-mri
# Module:    $RCSfile: mhd_utils.py,v $
# Language:  Python
# Author:    $Author: bjian $
# Date:      $Date: 2008/10/27 05:55:55 $
# Version:
#           $Revision: 1.1 by PJackson 2013/06/06 $
#               Modification: Adapted to 3D
#               Link: https://sites.google.com/site/pjmedphys/tutorials/medical-images-in-python
#
#           $Revision: 2   by RodenLuo 2017/03/12 $
#               Modication: Adapted to LUNA2016 data set for DSB2017
#               Link:https://www.kaggle.com/rodenluo/data-science-bowl-2017/crop-save-and-view-nodules-in-3d
#           $Revision: 3   by Sober.JChen 2017/03/15 $
#               Modication: Adapted to LUNA2016 data set for DSB2017
#               Link:
#-------write_meta_header,dump_raw_data,write_mhd_file------三个函数的由来
# ======================================================================

import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import os
import array
import math
import random
import shutil
from toolbox import MITools as mt
from toolbox import CTViewer_Multiax as cvm
from toolbox import Lung_Cluster as lc
from toolbox import CandidateDetection as cd
try:
	from tqdm import tqdm # long waits are not fun
except:
	print('')
	tqdm = lambda x : x
# import traceback

BOX_SIZE = 56

class NodulesCrop(object):
	def __init__(self, all_patients_path, annotations_file, output_path, vision_path):
		"""param: workspace: 本次比赛all_patients的父目录"""
		self.all_patients_path = all_patients_path
		self.annotations_file = annotations_file
		self.output_path = output_path
		self.vision_path = vision_path
		self.nodules_npy_path = output_path + "npy/"
		self.nonnodule_npy_path = output_path + "npy_non/"

	def save_annotations_nodule(self, nodule_crop, store_name, mhd_store=False):
		np.save(os.path.join(self.nodules_npy_path, store_name + "_annotations.npy"), nodule_crop)
		if mhd_store:
			mt.write_mhd_file(self.all_annotations_mhd_path + store_name + "_annotations.mhd", nodule_crop, nodule_crop.shape)

	def save_nonnodule(self, nodule_crop, store_name, mhd_store=False):
		np.save(os.path.join(self.nonnodule_npy_path, store_name + "_nonannotation.npy"), nodule_crop)
		if mhd_store:
			mt.write_mhd_file(self.no_annotation_mhd_path + store_name + "_nonannotation.mhd", nodule_crop, nodule_crop.shape)

	def get_filename(self,file_list, case):
		for f in file_list:
			if case in f:
				return (f)

	def get_ct_constants(self):
		maxvalue = -2000
		minvalue = 2000
		for patient in enumerate(tqdm(self.ls_all_patients)):
			patient = patient[1]
			#print(patient)
			patient_uid = mt.get_serie_uid(patient)
			patient_nodules = self.df_annotations[self.df_annotations.file == patient]
			full_image_info = sitk.ReadImage(patient)
			full_scan = sitk.GetArrayFromImage(full_image_info)
			full_scan[full_scan<-1024] = -1024
			segimage, segmask, flag = cd.segment_lung_mask(full_scan)
			vmax = full_scan[segmask==1].max()
			vmin = full_scan[segmask==1].min()
			if maxvalue<vmax:
				maxvalue = vmax
				maxfile = patient
			if minvalue>vmin:
				minvalue = vmin
		print("maxvalue:%d minvalue:%d" %(maxvalue, minvalue))
		print("%s" %(maxfile))
		return maxvalue, minvalue, maxfile

class NodulesCropMhd(NodulesCrop):
	def __init__(self, workspace="./", all_patients_path="./sample_patients/", annotations_file="./csv_files/annotations.csv", output_path="./nodule_cubes/", vision_path="./detection_vision"):
		"""param: workspace: 本次比赛all_patients的父目录"""
		self.workspace = workspace
		self.all_patients_path = all_patients_path
		self.annotations_file = annotations_file
		self.output_path = output_path
		self.vision_path = vision_path
		self.nodules_npy_path = output_path + "npy/"
		self.all_annotations_mhd_path = output_path + "mhd/"
		self.nonnodule_npy_path = output_path + "npy_non/"
		self.no_annotation_mhd_path = output_path + "mhd_non/"
		#self.all_candidates_mhd_path = output_path + "mhd_random/"
		self.ls_all_patients = glob(self.all_patients_path + "*.mhd")
		self.df_annotations = pd.read_csv(self.annotations_file)
		self.df_annotations["file"] = self.df_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
		self.df_annotations = self.df_annotations.dropna()

	def annotations_crop(self, randsample=True, candsample=False, overbound=False, augment=False):#the term 'augment' is invalid when 'overbound' is True
		if os.access(self.output_path, os.F_OK):
			shutil.rmtree(self.output_path)
		os.makedirs(self.output_path)
		os.mkdir(self.nodules_npy_path)	#训练用正样本路径
		os.mkdir(self.all_annotations_mhd_path)	#检查用正样本路径
		os.mkdir(self.nonnodule_npy_path)	#训练用负样本路径
		os.mkdir(self.no_annotation_mhd_path)	#检查用负样本路径

		if not os.access(self.vision_path, os.F_OK):
			os.makedirs(self.vision_path)

		for patient in enumerate(tqdm(self.ls_all_patients)):
			patient = patient[1]
			#patient = './LUNA16/subset9\\1.3.6.1.4.1.14519.5.2.1.6279.6001.114914167428485563471327801935.mhd'
			print(patient)
			# 检查这个病人有没有大于3mm的结节标注
			if patient not in self.df_annotations.file.values:
				print('Patient ' + patient + 'Not exist!')
				continue
			patient_uid = mt.get_serie_uid(patient)
			patient_nodules = self.df_annotations[self.df_annotations.file == patient]
			full_image_info = sitk.ReadImage(patient)
			full_scan = sitk.GetArrayFromImage(full_image_info)
			origin = np.array(full_image_info.GetOrigin())[::-1]  #---获取“体素空间”中结节中心的坐标
			old_spacing = np.array(full_image_info.GetSpacing())[::-1]  #---该CT在“世界空间”中各个方向上相邻单位的体素的间距
			image, new_spacing = mt.resample(full_scan, old_spacing)#---重采样
			print('resample done')
			v_centers = []
			center_coords = []
			for index, nodule in patient_nodules.iterrows():
				nodule_diameter = nodule.diameter_mm
				nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])#---获取“世界空间”中结节中心的坐标
				v_center = np.rint((nodule_center - origin) / new_spacing)#映射到“体素空间”中的坐标
				v_center = np.array(v_center, dtype=int)
				v_centers.append([index, nodule_diameter, v_center])
				center_coords.append(v_center)
			#volume_regioned = cvm.view_coordinations(image, center_coords, window_size=int(math.ceil(1.5*nodule_diameter)), reverse=False, slicewise=False, show=False)
			#np.save(self.vision_path+"/"+patient_uid+"_annotated.npy", volume_regioned)
			#---这一系列的if语句是根据“判断一个结节的癌性与否需要结合该结节周边位置的阴影和位置信息”而来，故每个结节都获取了比该结节尺寸略大的3D体素

			#get annotations nodule
			window_half = int(BOX_SIZE/2)
			if overbound:
				num_translations = 1
				for index, nodule_diameter, v_center in v_centers:
					zyx_1 = v_center - BOX_SIZE  # 注意是: Z, Y, X
					zyx_2 = v_center + BOX_SIZE
					if mt.coord_overflow(zyx_1, image.shape) or mt.coord_overflow(zyx_2, image.shape):
						continue
					nodule_box = np.zeros([2*BOX_SIZE, 2*BOX_SIZE, 2*BOX_SIZE], np.int16)  # ---nodule_box_size = 45
					img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  # ---截取立方体
					img_crop[img_crop<-1024] = -1024  # ---设置窗宽，小于-1024的体素值设置为-1024
					try:
						nodule_box = img_crop[0:2*BOX_SIZE, 0:2*BOX_SIZE, 0:2*BOX_SIZE]  # ---将截取的立方体置于nodule_box
					except:
						print("annotation error")
						continue
					#nodule_box[nodule_box == 0] = -1024  # ---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
					self.save_annotations_nodule(nodule_box, patient_uid+"_"+str(index)+"_ob")
			else:
				if not augment:
					scales = [1.0]
					translations = np.array([0,0,0])
				else:
					scales = [0.8,1.0,1.25]
					#translations = np.array([[0,0,0],[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]], dtype=float)
					translations = np.array([[0,0,0],[0,0,1],[0,0,-1], [0,1,0],[0,math.sqrt(0.5),math.sqrt(0.5)],[0,math.sqrt(0.5),-math.sqrt(0.5)], [0,-1,0],[0,-math.sqrt(0.5),math.sqrt(0.5)],[0,-math.sqrt(0.5),-math.sqrt(0.5)],
							    [1,0,0],[math.sqrt(0.5),0,math.sqrt(0.5)],[math.sqrt(0.5),0,-math.sqrt(0.5)], [math.sqrt(0.5),math.sqrt(0.5),0],[math.sqrt(0.3333),math.sqrt(0.3333),math.sqrt(0.3333)],[math.sqrt(0.3333),math.sqrt(0.3333),-math.sqrt(0.3333)], [math.sqrt(0.5),-math.sqrt(0.5),0],[math.sqrt(0.3333),-math.sqrt(0.3333),math.sqrt(0.3333)],[math.sqrt(0.3333),-math.sqrt(0.3333),-math.sqrt(0.3333)],
							    [-1,0,0],[-math.sqrt(0.5),0,math.sqrt(0.5)],[-math.sqrt(0.5),0,-math.sqrt(0.5)], [-math.sqrt(0.5),math.sqrt(0.5),0],[-math.sqrt(0.3333),math.sqrt(0.3333),math.sqrt(0.3333)],[-math.sqrt(0.3333),math.sqrt(0.3333),-math.sqrt(0.3333)], [-math.sqrt(0.5),-math.sqrt(0.5),0],[-math.sqrt(0.3333),-math.sqrt(0.3333),math.sqrt(0.3333)],[-math.sqrt(0.3333),-math.sqrt(0.3333),-math.sqrt(0.3333)]])

				num_translations = 3
				for index, nodule_diameter, v_center in v_centers:
					for s in range(len(scales)):
						rt = np.zeros(num_translations, dtype=int)
						rt[1:num_translations] = np.random.choice(range(1,len(translations)), num_translations-1, False)
						rt = np.sort(rt)
						for t in range(rt.size):
							scale = scales[s]
							box_size = int(np.ceil(BOX_SIZE*scale))
							window_size = int(box_size/2)
							translation = np.array(nodule_diameter/2*translations[rt[t]]/new_spacing, dtype=int)
							tnz = translation.nonzero()
							if tnz[0].size==0 and t!=0:
								continue
							zyx_1 = v_center + translation - window_size  # 注意是: Z, Y, X
							zyx_2 = v_center + translation + box_size - window_size
							if mt.coord_overflow(zyx_1, image.shape) or mt.coord_overflow(zyx_2, image.shape):
								continue
							nodule_box = np.zeros([BOX_SIZE, BOX_SIZE, BOX_SIZE], np.int16)  # ---nodule_box_size = 45
							img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  # ---截取立方体
							img_crop[img_crop<-1024] = -1024  # ---设置窗宽，小于-1024的体素值设置为-1024
							if not augment or scale==1.0:
								img_crop_rescaled = img_crop
							else:
								img_crop_rescaled, rescaled_spacing = mt.resample(img_crop, new_spacing, new_spacing*scale)
							try:
								padding_shape = (img_crop_rescaled.shape - np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE])) / 2
								nodule_box = img_crop_rescaled[padding_shape[0]:padding_shape[0]+BOX_SIZE, padding_shape[1]:padding_shape[1]+BOX_SIZE, padding_shape[2]:padding_shape[2]+BOX_SIZE]  # ---将截取的立方体置于nodule_box
							except:
								# f = open("log.txt", 'a')
								# traceback.print_exc(file=f)
								# f.flush()
								# f.close()
								print("annotation error")
								continue
							#nodule_box[nodule_box == 0] = -1024  # ---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
							self.save_annotations_nodule(nodule_box, patient_uid+"_"+str(index)+"_"+str(s*rt.size+t))
			print("annotation sampling done")

			#get candidate annotation nodule
			candidate_coords = []
			if candsample:
				segimage, segmask, flag = cd.segment_lung_mask(image)
				if segimage is not None:
					#nodule_matrix, index = cd.candidate_detection(segimage,flag)
					#cluster_labels = lc.seed_mask_cluster(nodule_matrix, cluster_size=1000)
					cluster_labels = lc.seed_volume_cluster(image, segmask, eliminate_lower_size=-1)
					segresult = lc.segment_color_vision(image, cluster_labels)
					cvm.view_CT(segresult)
					#lc.cluster_size_vision(cluster_labels)
					exit()
					candidate_coords, _ = lc.cluster_centers(cluster_labels)
					#candidate_coords = lc.cluster_center_filter(image, candidate_coords)
				#the coordination order is [z,y,x]
				print("candidate number:%d" %(len(candidate_coords)))
				#volume_regioned = cv.view_coordinations(image, candidate_coords, window_size=10, reverse=False, slicewise=True, show=False)
				#mt.write_mhd_file(self.vision_path+"/"+patient_uid+"_candidate.mhd", volume_regioned, volume_regioned.shape[::-1])
				for cc in range(len(candidate_coords)):
					candidate_center = candidate_coords[cc]
					invalid_loc = False
					if mt.coord_overflow(candidate_center-window_half, image.shape) or mt.coord_overflow(candidate_center+BOX_SIZE-window_half, image.shape):
						invalid_loc = True
						continue
					for index_search, nodule_diameter_search, v_center_search in v_centers:
						rpos = v_center_search - candidate_center
						if abs(rpos[0])<window_half and abs(rpos[1])<window_half and abs(rpos[2])<window_half:  #the negative sample is located in the positive location
							invalid_loc = True
							break
					if not invalid_loc:
						zyx_1 = candidate_center - window_half
						zyx_2 = candidate_center + BOX_SIZE - window_half
						nodule_box = np.zeros([BOX_SIZE,BOX_SIZE,BOX_SIZE], np.int16)#---nodule_box_size = 45
						img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]#---截取立方体
						img_crop[img_crop<-1024] = -1024	#---设置窗宽，小于-1000的体素值设置为-1000
						if img_crop.shape[0]!=BOX_SIZE | img_crop.shape[1]!=BOX_SIZE | img_crop.shape[2]!=BOX_SIZE:
							print("error in resmapleing shape")
						try:
							nodule_box[0:BOX_SIZE, 0:BOX_SIZE, 0:BOX_SIZE] = img_crop  # ---将截取的立方体置于nodule_box
						except:
							print("random error")
							continue
						#nodule_box[nodule_box == 0] = -1024#---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
						self.save_nonnodule(nodule_box, patient_uid+"_cc_"+str(cc))
				print("candidate sampling done")

			#get random annotation nodule
			if randsample:
				if overbound:
					augnum = 100
				elif augment:
					augnum = len(scales) * num_translations
				else:
					augnum = 1
				if augnum*len(v_centers)>len(candidate_coords):
					randnum = augnum*len(v_centers) - len(candidate_coords)
				else:
					randnum = len(candidate_coords)
				for rc in range(randnum):  #the random samples is one-to-one number of nodules
					#index, nodule_diameter, v_center = v_centers[rc]
					rand_center = np.array([0,0,0])  # 注意是: Z, Y, X
					invalid_loc = True
					candidate_overlap = True
					while invalid_loc:
						invalid_loc = False
						candidate_overlap = False
						for axis in range(rand_center.size):
							rand_center[axis] = np.random.randint(0, image.shape[axis])
						if mt.coord_overflow(rand_center-window_half, image.shape) or mt.coord_overflow(rand_center+BOX_SIZE-window_half, image.shape):
							invalid_loc = True
							continue
						if 'segmask' in dir() and not (segmask is None) and not segmask[rand_center[0], rand_center[1], rand_center[2]]:
							invalid_loc = True
							continue
						for index_search, nodule_diameter_search, v_center_search in v_centers:
							rpos = v_center_search - rand_center
							if abs(rpos[0])<window_half and abs(rpos[1])<window_half and abs(rpos[2])<window_half:  #the negative sample is located in the positive location
								invalid_loc = True
								break
						for candidate_coord in candidate_coords:
							rpos = candidate_coord - rand_center
							if abs(rpos[0])<window_half and abs(rpos[1])<window_half and abs(rpos[2])<window_half:  #the negative sample is located in the pre-extracted candidate locations
								candidate_overlap = True
								break
					if candidate_overlap:
						continue
                        
					zyx_1 = rand_center - window_half
					zyx_2 = rand_center + BOX_SIZE - window_half
					nodule_box = np.zeros([BOX_SIZE,BOX_SIZE,BOX_SIZE],np.int16)#---nodule_box_size = 45
					img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]#---截取立方体
					img_crop[img_crop<-1024] = -1024	#---设置窗宽，小于-1000的体素值设置为-1000
					if img_crop.shape[0]!=BOX_SIZE | img_crop.shape[1]!=BOX_SIZE | img_crop.shape[2]!=BOX_SIZE:
						print("error in resmapleing shape")
					try:
						nodule_box[0:BOX_SIZE, 0:BOX_SIZE, 0:BOX_SIZE] = img_crop  # ---将截取的立方体置于nodule_box
					except:
						# f = open("log.txt", 'a')
						# traceback.print_exc(file=f)
						# f.flush()
						# f.close()
						print("candidate error")
						continue
					#nodule_box[nodule_box == 0] = -1024#---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
					self.save_nonnodule(nodule_box, patient_uid+"_rc_"+str(rc))
				print("random sampling done")

			print('Done for this patient!\n\n')
		print('Done for all!')

	def candidates_crop(self):	#the term 'augment' is invalid when 'overbound' is True
		if os.access(self.output_path, os.F_OK):
			shutil.rmtree(self.output_path)
		os.makedirs(self.output_path)
		os.mkdir(self.nodules_npy_path)		#训练用正样本路径
		os.mkdir(self.nonnodule_npy_path)	#训练用负样本路径

		if not os.access(self.vision_path, os.F_OK):
			os.makedirs(self.vision_path)

		for patient in enumerate(tqdm(self.ls_all_patients)):
			patient = patient[1]
			#patient = './TIANCHI_data/val/LKDS-00002.mhd'
			print(patient)
			# 检查这个病人有没有大于3mm的结节
			if patient not in self.df_annotations.file.values:
				print('Patient ' + patient + 'Not exist!')
				continue
			patient_uid = mt.get_serie_uid(patient)
			patient_nodules = self.df_annotations[self.df_annotations.file == patient]
			full_image_info = sitk.ReadImage(patient)
			full_scan = sitk.GetArrayFromImage(full_image_info)
			origin = np.array(full_image_info.GetOrigin())[::-1]  #---获取“体素空间”中结节中心的坐标
			old_spacing = np.array(full_image_info.GetSpacing())[::-1]  #---该CT在“世界空间”中各个方向上相邻单位的体素的间距
			image, new_spacing = mt.resample(full_scan, old_spacing)#---重采样
			print('resample done')
			v_centers = []
			nonnodule_coords = []
			nodule_coords = []
			for index, nodule in patient_nodules.iterrows():
				nodule_class = nodule.get("class")
				nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])#---获取“世界空间”中结节中心的坐标
				v_center = np.rint((nodule_center - origin) / new_spacing)#映射到“体素空间”中的坐标
				v_center = np.array(v_center, dtype=int)
				v_centers.append([index, nodule_class, v_center])
				if nodule_class==1:
					nodule_coords.append(v_center)
				else:
					nonnodule_coords.append(v_center)
			#volume_regioned = cvm.view_coordinations(image, nonnodule_coords, window_size=56, reverse=False, slicewise=True, show=False)
			#np.save(self.vision_path+"/"+patient_uid+"_candidatenonnodule.npy", volume_regioned)
			#volume_regioned = cvm.view_coordinations(image, nodule_coords, window_size=10, reverse=False,
			#					 slicewise=False, show=False)
			#np.save(self.vision_path+"/"+patient_uid+"_candidatenodule.npy", volume_regioned)
			#---这一系列的if语句是根据“判断一个结节的癌性与否需要结合该结节周边位置的阴影和位置信息”而来，故每个结节都获取了比该结节尺寸略大的3D体素

			#get annotations nodule
			window_half = int(BOX_SIZE/2)
			num_translations = 1
			for index, nodule_class, v_center in v_centers:
				invalid_loc = False
				if nodule_class==0:
					for nodule_coord in nodule_coords:
						rpos = nodule_coord - v_center
						if abs(rpos[0]) <= window_half and abs(rpos[1]) <= window_half and abs(rpos[2]) <= window_half:
							# the negative sample is located in the positive location
							invalid_loc = True
							break
				if not invalid_loc:
					zyx_1 = v_center - window_half  # 注意是: Z, Y, X
					zyx_2 = v_center + BOX_SIZE - window_half
					if mt.coord_overflow(zyx_1, image.shape) or mt.coord_overflow(zyx_2, image.shape):
						continue
					nodule_box = np.zeros([BOX_SIZE, BOX_SIZE, BOX_SIZE], np.int16)  # ---nodule_box_size = 45
					img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  # ---截取立方体
					img_crop[img_crop<-1024] = -1024  # ---设置窗宽，小于-1024的体素值设置为-1024
					try:
						nodule_box = img_crop[0:BOX_SIZE, 0:BOX_SIZE, 0:BOX_SIZE]  # ---将截取的立方体置于nodule_box
					except:
						print("annotation error")
						continue
					if nodule_class==0:
						self.save_nonnodule(nodule_box, patient_uid+"_"+str(index)+"_cc")
					#else:
					#	self.save_annotations_nodule(nodule_box, patient_uid+"_"+str(index)+"_ob")
			print('Done for this patient!\n\n')
		print('Done for all!')


class SLHCrop(NodulesCrop):
	def __init__(self, all_patients_path="./SLH_data/", annotations_file="./csv_files/annotations.csv", output_path="./nodule_cubes/", vision_path="./detection_vision"):
		"""param: workspace: 本次比赛all_patients的父目录"""
		self.all_patients_path = all_patients_path
		self.annotations_file = annotations_file
		self.output_path = output_path
		self.vision_path = vision_path
		self.nodules_npy_path = output_path + "npy/"
		self.nonnodule_npy_path = output_path + "npy_non/"	
		self.df_annotations = pd.read_excel(self.annotations_file)
		self.annotation_columns = ["anno1", "anno2", "anno3", "anno4", "anno5", "anno6", "anno7", "anno8", "anno9"]
		self.ls_all_patients = []
		time_packages = mt.get_dirs(self.all_patients_path)
		for package in time_packages:
			self.ls_all_patients.extend(mt.get_dirs(package))

	def annotations_crop(self, overbound = True, candsample = False):
		if os.access(self.output_path, os.F_OK):
			shutil.rmtree(self.output_path)
		os.makedirs(self.output_path)
		os.mkdir(self.nodules_npy_path)
		os.mkdir(self.nonnodule_npy_path)

		if not os.access(self.vision_path, os.F_OK):
			os.makedirs(self.vision_path)

		for patient in enumerate(tqdm(self.ls_all_patients[814:])):
			patient = patient[1]
			#patient = "./SLH_data/0721/285 0800418645"
			print(patient)
			full_scan, full_image_info, patient_uid = mt.read_dicom_scan(patient)
			if full_scan.min()<-1024:
				errorlog = open("results/error.log", "w")
				errorlog.write("Hu unit incorrect:%s\n" %(patient))
				errorlog.close()
			origin = np.array(full_image_info.GetOrigin())[::-1]  #---获取“体素空间”中结节中心的坐标
			old_spacing = np.array(full_image_info.GetSpacing())[::-1]  #---该CT在“世界空间”中各个方向上相邻单位的体素的间距
			min_space = old_spacing.min()
			image, new_spacing = mt.resample(full_scan, old_spacing) #---重采样
			print('resample done')

			silist = self.df_annotations.serie_id.tolist()
			if silist.count(patient_uid)==0:
				print('no annotation for this patient found')
				continue
			serie_index = silist.index(patient_uid)
			patient_nodules = []
			for annocol in self.annotation_columns:
				annostr = self.df_annotations.get(annocol)[serie_index]
				if type(annostr)==unicode:
					#annotation = np.array(annostr.split(u'\uff08')[0].split(' '), dtype=int)
					#patient_nodules.append([serie_index, annotation]) #the index order is [x,y,z]
					if annostr.find(u'*')>=0:
						continue
					coordbegin = -1
					coordend = -1
					for ci in range(len(annostr)):
						if coordbegin<0:
							if annostr[ci]>=u'0' and annostr[ci]<=u'9':
								coordbegin = ci
						elif (annostr[ci]<u'0' or annostr[ci]>u'9') and annostr[ci]!=u' ':
							coordend = ci
							break
					if coordbegin>=0:
						if coordend<0:
							coordend = len(annostr)
						coordstr = annostr[coordbegin:coordend]
						annotation = np.array(coordstr.split(u' '), dtype=int)
						patient_nodules.append([annocol, annotation])  # the index order is [x,y,z]
				if type(annostr)==str:
					if annostr.find('*')>=0:
						continue
					coordbegin = -1
					coordend = -1
					for ci in range(len(annostr)):
						if coordbegin<0:
							if annostr[ci]>='0' and annostr[ci]<='9':
								coordbegin = ci
						elif (annostr[ci]<'0' or annostr[ci]>'9') and annostr[ci]!=' ':
							coordend = ci
							break
					if coordbegin>=0:
						# annotation = np.array(annostr.split('（')[0].split(' '), dtype=int)
						if coordend<0:
							coordend = len(annostr)
						coordstr = annostr[coordbegin:coordend]
						annotation = np.array(coordstr.split(' '), dtype=int)
						patient_nodules.append([annocol, annotation])  # the index order is [x,y,z]
			
			v_centers = []
			center_coords = []
			for annocol, nodule in patient_nodules:
				nodule_center = np.array(np.flip(nodule, axis=0)*old_spacing/new_spacing, dtype=int) #---获取“世界空间”中结节中心的坐标
				#v_center = np.rint((nodule_center - origin) / new_spacing) #映射到“体素空间”中的坐标
				#v_center = np.array(v_center, dtype=int)
				v_centers.append([annocol, nodule_center])
				center_coords.append(nodule_center)
			#volume_regioned = cv.view_coordinations(image, center_coords, window_size=10, reverse=False, slicewise=False, show=False)
			#cv.view_CT(volume_regioned)
			#np.save(self.vision_path+"/"+patient_uid+"_annotated.mhd", volume_regioned)
			#---这一系列的if语句是根据“判断一个结节的癌性与否需要结合该结节周边位置的阴影和位置信息”而来，故每个结节都获取了比该结节尺寸略大的3D体素

			#get annotations nodule
			window_half = int(BOX_SIZE/2)
			if overbound:
				box_size = 2 * BOX_SIZE
				box_half = BOX_SIZE
			else:
				box_size = BOX_SIZE
				box_half = window_half

			for annocol, v_center in v_centers:
				zyx_1 = v_center - box_half  # 注意是: Z, Y, X
				zyx_2 = v_center + box_half
				if mt.coord_overflow(zyx_1, image.shape) or mt.coord_overflow(zyx_2, image.shape):
					zyx_1_fix = zyx_1.copy()
					zyx_2_fix = zyx_2.copy()
					for ci in range(3):
						if zyx_1[ci] < 0:
							zyx_1_fix[ci] = 0
						elif zyx_1[ci] >= image.shape[ci]:
							zyx_1_fix[ci] = image.shape[ci]
						if zyx_2[ci] < 0:
							zyx_2_fix[ci] = 0
						elif zyx_2[ci] >= image.shape[ci]:
							zyx_2_fix[ci] = image.shape[ci]
					img_crop = image[zyx_1_fix[0]:zyx_2_fix[0], zyx_1_fix[1]:zyx_2_fix[1], zyx_1_fix[2]:zyx_2_fix[2]]
					img_crop[img_crop<-1024] = -1024
					#if img_crop.max() >= 600:
					#	padding_value = 600
					#elif img_crop.max() >= 0:
					#	padding_value = img_crop.max()
					#else:
					#	padding_value = -1024
					padding_value = -1024
					nodule_box = padding_value * np.ones([box_size, box_size, box_size], int)
					nodule_box[zyx_1_fix[0]-zyx_1[0]:zyx_2_fix[0]-zyx_1[0], zyx_1_fix[1]-zyx_1[1]:zyx_2_fix[1]-zyx_1[1], zyx_1_fix[2]-zyx_1[2]:zyx_2_fix[2]-zyx_1[2]] = img_crop
				else:
					#nodule_box = np.zeros([box_size, box_size, box_size], np.int16)
					img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  # ---截取立方体
					img_crop[img_crop<-1024] = -1024  # ---设置窗宽，小于-1024的体素值设置为-1024
					nodule_box = img_crop[0:box_size, 0:box_size, 0:box_size]  # ---将截取的立方体置于nodule_box
				self.save_annotations_nodule(nodule_box, patient_uid+"_"+annocol+"_ob")
			print("annotation sampling done")

			#get candidate annotation nodule
			candidate_coords = []
			if candsample:
				segimage, segmask, flag = cd.segment_lung_mask(image)
				if segimage is not None:
					nodule_matrix, index = cd.candidate_detection(segimage,flag)
					cluster_labels = lc.seed_mask_cluster(nodule_matrix, cluster_size=1000)
					#cluster_labels = lc.seed_volume_cluster(image, segmask, eliminate_lower_size=-1)
					#segresult = lc.segment_color_vision(image, cluster_labels)
					#cv.view_CT(segresult)
					#lc.cluster_size_vision(cluster_labels)
					candidate_coords, _ = lc.cluster_centers(cluster_labels)
					#candidate_coords = lc.cluster_center_filter(image, candidate_coords)
				#the coordination order is [z,y,x]
				print("candidate number:%d" %(len(candidate_coords)))
				#volume_regioned = cv.view_coordinations(image, candidate_coords, window_size=10, reverse=False, slicewise=True, show=False)
				#mt.write_mhd_file(self.vision_path+"/"+patient_uid+"_candidate.mhd", volume_regioned, volume_regioned.shape[::-1])
				for cc in range(len(candidate_coords)):
					candidate_center = candidate_coords[cc]
					invalid_loc = False
					if mt.coord_overflow(candidate_center-window_half, image.shape) or mt.coord_overflow(candidate_center+BOX_SIZE-window_half, image.shape):
						invalid_loc = True
						continue
					for index_search, v_center_search in v_centers:
						rpos = v_center_search - candidate_center
						if abs(rpos[0])<window_half and abs(rpos[1])<window_half and abs(rpos[2])<window_half:  #the negative sample is located in the positive location
							invalid_loc = True
							break
					if not invalid_loc:
						zyx_1 = candidate_center - window_half
						zyx_2 = candidate_center + BOX_SIZE - window_half
						nodule_box = np.zeros([BOX_SIZE,BOX_SIZE,BOX_SIZE], np.int16)	#---nodule_box_size = 45
						img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]	#---截取立方体
						img_crop[img_crop<-1024] = -1024	#---设置窗宽，小于-1000的体素值设置为-1000
						if img_crop.shape[0]!=BOX_SIZE | img_crop.shape[1]!=BOX_SIZE | img_crop.shape[2]!=BOX_SIZE:
							print("error in resmapleing shape")
						try:
							nodule_box[0:BOX_SIZE, 0:BOX_SIZE, 0:BOX_SIZE] = img_crop  # ---将截取的立方体置于nodule_box
						except:
							print("random error")
							continue
						#nodule_box[nodule_box == 0] = -1024#---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
						self.save_nonnodule(nodule_box, patient_uid+"_cc_"+str(cc))
				print("candidate sampling done")
			print('Done for this patient!\n\n')
		print('Done for all!')

if __name__ == '__main__':
	subsets = ['subset0', 'subset1', 'subset2', 'subset3', 'subset4', 'subset5', 'subset6', 'subset7', 'subset8', 'subset9']
	#subsets = ['subset0']
	for subset in subsets:
		#nodule_cube_subset = NodulesCropMhd("./", "./LUNA16/"+subset+"/", "./LUNA16/csvfiles/annotations.csv", "./nodule_cubes/"+subset+"/")
		#nodule_cube_subset.annotations_crop(overbound=True, candsample=False, randsample=False)
		nodule_cube_subset = NodulesCropMhd("./", "./LUNA16/"+subset+"/", "./LUNA16/csvfiles/candidates.csv", "./nodule_cubes/"+subset+"/")
		nodule_cube_subset.candidates_crop()
	#nc_train = NodulesCropMhd("./", "./TIANCHI_data/train/", "./TIANCHI_data/csv_files/train/annotations.csv", "./nodule_cubes/train/", "./detection_vision/train/")
	#nc_train.annotations_crop(overbound=True, candsample=False, randsample=False)
	#nc_val = NodulesCropMhd("./", "./TIANCHI_data/val/", "./TIANCHI_data/csv_files/val/annotations.csv", "./nodule_cubes/val/", "./detection_vision/val")
	#nc_val.annotations_crop(overbound=True, candsample=False, randsample=False)

	#sc = SLHCrop("./SLH_data/", "./SLH_data/annotations.xlsx", "./nodule_cubes/")
	#sc.annotations_crop(overbound=True, candsample=False)
