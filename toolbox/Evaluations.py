import os
import copy
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
	from tqdm import tqdm  # long waits are not fun
except:
	print('tqdm not installed')
	tqdm = lambda x: x

def detection_assessment(results_input, annofile, candfile = None):
	if len(results_input)==0:
		print('no results to assess')
		return None
	results = copy.deepcopy(results_input)
	true_positive = 0
	false_positive = 0
	annotated_uids = []
	TPwithFPs = []
	FPsperscan = []
	sensitivities = []
	sensitivities2 = []

	CPMscore = 0
	CPMscore2 = 0
	targetFPsperscan = [0.125, 0.25, 0.5, 1, 2, 4, 8]
	#fpind = 0

	annotations = pd.read_csv(annofile)
	anno_assessed = np.zeros(shape=len(annotations["seriesuid"].values), dtype=bool)
	anno_detected = np.zeros(shape=len(annotations["seriesuid"].values), dtype=bool)
	if candfile is not None:
		candidates = pd.read_csv(candfile)

	'''
	#add a true detection for testing
	results = copy.deepcopy(results_input)
	num_results = len(results)
	uids = []
	for ri in range(num_results):
		result = results[ri]
		uid = result[0]
		if uids.count(uid)==0:
			uids.append(uid)
			annolines = (annotations["seriesuid"].values == uid).nonzero()[0]
			for annoline in annolines:
				anno_assessed[annoline] = True
				for i in range(2):	#insert two true result
					annoX = float(annotations["coordX"].values[annoline]) + random.random()
					annoY = float(annotations["coordY"].values[annoline]) + random.random()
					annoZ = float(annotations["coordZ"].values[annoline]) + random.random()
					results.append([uid, annoX, annoY, annoZ, random.random()])
	'''

	predictions = np.zeros(shape=len(results), dtype=float)
	for r in range(len(results)):
		predictions[r] = results[r][4]
	confsort = predictions.argsort()
	confsort = confsort[::-1]
	nodules_detected = 0 - np.ones(shape=len(confsort), dtype=int)
	for ci in range(len(confsort)):
		result = results[confsort[ci]]
		uid = result[0]
		coord = result[1:4]
		annolines = (annotations["seriesuid"].values == uid).nonzero()[0]
		if len(annolines)>0 and annotated_uids.count(uid)==0:
			annotated_uids.append(uid)
		truenodule = False
		nearestanno = -1
		minsquaredist = 2000
		for annoline in annolines:
			anno_assessed[annoline] = True
			annoX = float(annotations["coordX"].values[annoline])
			annoY = float(annotations["coordY"].values[annoline])
			annoZ = float(annotations["coordZ"].values[annoline])
			annodiam =  float(annotations["diameter_mm"].values[annoline])
			if abs(coord[0]-annoX) <= annodiam/2 and abs(coord[1]-annoY) <= annodiam/2 and abs(coord[2]-annoZ) <= annodiam/2:
				#assuming that the annotations do not intersect with each other
				truenodule = True
				squaredist = (coord[0] - annoX) * (coord[0] - annoX) + (coord[1] - annoY) * (coord[1] - annoY) + (coord[2] - annoZ) * (coord[2] - annoZ)
				if minsquaredist>squaredist:
					minsquaredist = squaredist
					nearestanno = annoline
		if nearestanno >= 0:
			anno_detected[nearestanno] = True

		if not truenodule:
			suspected = False
			if 'candidates' in dir():
				candlines = np.logical_and(candidates["seriesuid"].values==uid, candidates["class"].values==1).nonzero()[0]
				for candline in candlines:
					candX = float(candidates["coordX"].values[candline])
					candY = float(candidates["coordY"].values[candline])
					candZ = float(candidates["coordZ"].values[candline])
					if abs(coord[0]-candX) <= 1.5 and abs(coord[1]-candY) <= 1.5 and abs(coord[2]-candZ) <= 1.5:
						print(uid + ":" + str(candline) + " ignore detecting coordinate")		#the coordinate detected is to suspected nodules, so we ignore it in evaluation.
						suspected = True
						break
			if suspected:
				nodules_detected[ci] = -2
			else:
				false_positive += 1
				#TPwithFPs.append([true_positive, false_positive])
				TPwithFPs.append([np.count_nonzero(anno_detected), false_positive])	#the number detected may be more than the number of annotations, thus we count the number of annotations detected as TT number
		else:
			nodules_detected[ci] = nearestanno
			true_positive += 1
	num_true = np.count_nonzero(anno_assessed)
	if num_true<=0:
		print('no real nodules for these scans')
		return None
	#calculate the self version of FROC parameters
	num_scans = len(annotated_uids)
	for true_positive, false_positive in TPwithFPs:
		fpord = 0
		for fpi in range(fpord, len(targetFPsperscan)):
			if false_positive == int(num_scans*targetFPsperscan[fpi]):
				#fpind += 1
				FPsperscan.append(targetFPsperscan[fpi])
				sensitivity = true_positive / float(num_true)
				sensitivities.append(sensitivity)
				CPMscore += sensitivity
				fpord = fpi + 1
			#if fpind>=len(targetFPsperscan):
			#	break
	if len(sensitivities)>0:
		CPMscore /= float(len(sensitivities))
	#calculate the stantard version of FROC parameters
	nodules_detected_nosuspected = nodules_detected[nodules_detected>=-1]
	for fpperscan in targetFPsperscan:
		scind = int(num_scans*fpperscan) + num_true
		if scind>0:
			noduleretrieve = nodules_detected_nosuspected[:scind]
			true_positive = np.unique(noduleretrieve[noduleretrieve>=0]).size
			sensitivity =  true_positive / float(num_true)
		else:
			sensitivity = 0
		sensitivities2.append(sensitivity)
		CPMscore2 += sensitivity
	if len(sensitivities2)>0:
		CPMscore2 /= float(len(sensitivities2))
	return num_scans, FPsperscan, sensitivities, CPMscore, targetFPsperscan, sensitivities2, CPMscore2, nodules_detected
	
def evaluation_vision(CPMs, num_scans, FPsperscan, sensitivities, CPMscore, nodules_detected, CPM_output, FROC_output):
	CPMs.append([CPMscore, sensitivities, num_scans])
	CPMoutput = open(CPM_output, "w")
	for CPM, sensitivity_list, num_scan in CPMs:
		CPMoutput.write("CPM:{} sensitivities:{} of {} scans\n" .format(CPM, sensitivity_list, num_scan))
	CPMoutput.write("detection order:\n{}" .format(nodules_detected))
	CPMoutput.close()
	print("CPM:{} sensitivities:{} of {} scans" .format(CPMscore, sensitivities, num_scans))
	if len(sensitivities)!=len(FPsperscan):
		print("axis incoorect")
		print("sensitivity:{}" .format(sensitivities))
		print("FPs number:{}" .format(FPsperscan))
	xaxis_range = [i for i in range(min(len(sensitivities), len(FPsperscan)))]
	plt.plot(xaxis_range, sensitivities[:len(xaxis_range)])
	plt.ylim(0, 1)
	plt.grid(True)
	plt.xlabel("FPs per scan")
	plt.ylabel("sensitivity")
	plt.xticks(xaxis_range, FPsperscan[:len(xaxis_range)])
	plt.savefig(FROC_output)
	plt.close()
	
def FROC_paint(FPsperscan_list, sensitivities_list, name_list, output_file):
	min_length = len(FPsperscan_list[0])
	for FPsperscan in FPsperscan_list:
		if min_length > len(FPsperscan):
			min_length = len(FPsperscan)
	for sensitivities in sensitivities_list:
		if min_length > len(sensitivities):
			min_length = len(sensitivities)
	color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	if len(color_list)<len(FPsperscan_list):
		print('No enough colors, partial exhibition')
		list_length = len(color_list)
	else:
		list_length = len(FPsperscan_list)
	lines = []
	for li in range(list_length):
		xaxis_range = [i for i in range(min_length)]
		line, = plt.plot(xaxis_range, sensitivities_list[li][:len(xaxis_range)], color=color_list[li])
		lines.append(line)
		plt.xticks(xaxis_range, FPsperscan_list[li][:len(xaxis_range)])
	plt.legend((lines[0], lines[1], lines[2], lines[3], lines[4]), (name_list[0], name_list[1], name_list[2], name_list[3], name_list[4]), loc="lower right")
	#plt.ylim(0, 1)
	plt.grid(True)
	plt.xlabel("Average number of false positives per scan")
	plt.ylabel("sensitivity")
	plt.savefig(output_file, format="pdf")
	plt.close()

def csv_evaluation(annotation_file, result_file, evaluation_path):
	if os.access(evaluation_path, os.F_OK):
		shutil.rmtree(evaluation_path)
	os.makedirs(evaluation_path)

	results = pd.read_csv(result_file)
	resultlist = []
	for r in range(len(results["seriesuid"].values)):
		uid = results["seriesuid"].values[r]
		coordX = results["coordX"].values[r]
		coordY = results["coordY"].values[r]
		coordZ = results["coordZ"].values[r]
		prob = results["probability"].values[r]
		resultlist.append([uid, coordX, coordY, coordZ, prob])

	assessment = detection_assessment(resultlist, annotation_file)
	if assessment is None:
		print('assessment failed')
		exit()
	num_scans, FPsperscan, sensitivities, CPMscore, FPsperscan2, sensitivities2, CPMscore2, nodules_detected = assessment

	if len(FPsperscan) <= 0 or len(sensitivities) <= 0:
		print("No results to evaluate, continue")
	else:
		evaluation_vision([], num_scans, FPsperscan, sensitivities, CPMscore, nodules_detected, CPM_file = evaluation_path + "/CPMscores.log", FROC_file = evaluation_path + "/froc_" + str(num_scans) + "scans.png")

	if len(FPsperscan2) <= 0 or len(sensitivities2) <= 0:
		print("No results to evaluate, continue")
	else:
		evaluation_vision([], num_scans, FPsperscan2, sensitivities2, CPMscore2, nodules_detected, CPM_file = evaluation_path + "/CPMscores2.log", FROC_file = evaluation_path + "/froc2_" + str(num_scans) + "scans.png")
	
if __name__ == "__main__":
	annotation_file = "../LUNA16/csvfiles/annotations.csv"
	result_files = ["../results/experiment1/result_20.csv",
			"../results/experiment1/evaluations_30/result.csv",
			"../results/experiment1/result_40.csv",
			"../results/experiment1/evaluation_committefusion/result.csv",
			"../results/experiment1/evaluation_latefusion/result.csv"]
	evaluation_path = "../results/evaluations_test"
	name_list = ['CNN-20','CNN-30','CNN-40','committe-fusion','late-fusion']
	FPsperscan_list = []
	sensitivities_list = []
	for result_file in result_files:
		results = pd.read_csv(result_file)
		resultlist = []
		for r in range(len(results["seriesuid"].values)):
			uid = results["seriesuid"].values[r]
			coordX = results["coordX"].values[r]
			coordY = results["coordY"].values[r]
			coordZ = results["coordZ"].values[r]
			prob = results["probability"].values[r]
			resultlist.append([uid, coordX, coordY, coordZ, prob])

		assessment = detection_assessment(resultlist, annotation_file)
		if assessment is None:
			print('{} assessment failed' .format(result_file))
			continue
		num_scans, FPsperscan, sensitivities, CPMscore, FPsperscan2, sensitivities2, CPMscore2, nodules_detected = assessment
		if len(FPsperscan) <= 0 or len(sensitivities) <= 0:
			print("No results to evaluate, continue")
		else:
			FPsperscan_list.append(FPsperscan2)
			sensitivities_list.append(sensitivities2)
	FROC_paint(FPsperscan_list, sensitivities_list, name_list, evaluation_path + "/froc_comparisons.pdf")