import os

def read_environment(envfile):
	ef = open(envfile)
	environment = ef.readlines()
	img_width = int(environment[0])
	img_height = int(environment[1])
	num_view = int(environment[2])
	max_bound = int(environment[3])
	min_bound = int(environment[4])
	pixel_mean = float(environment[5])
	ef.close()
	return img_width, img_height, num_view, max_bound, min_bound, pixel_mean

def read_constants(consfile):
	cf = open(consfile)
	constant_dictionary = {}
	parameters = cf.readlines()
	for parameter in parameters:
		parameter_split = parameter.split(':')
		if parameter_split[1].find('.')>=0:
			constant_dictionary[parameter_split[0]] = float(parameter_split[1])
		else:
			constant_dictionary[parameter_split[0]] = int(parameter_split[1])
	return constant_dictionary

def get_dirs(path):
	dirs = []
	entries = os.listdir(path)
	for entry in entries:
		entry_path = os.path.join(path, entry)
		if os.path.isdir(entry_path):
			dirs.append(entry_path)
	return dirs
	
def filelist_store(filelist, name):
	filelist_storage = open(name, "w")
	for file in filelist:
		filelist_storage.write("%s\n" %(file))
	filelist_storage.close()

def filelist_load(name):
	filelist_load = open(name, "r")
	filelist = filelist_load.readlines()
	for fi in range(len(filelist)):
		filelist[fi] = filelist[fi][:-1]
	filelist_load.close()
	return filelist
	
def dict2list(indict):
	outlist = []
	for key in indict.keys():
		outlist.append(indict[key])
	
	return outlist