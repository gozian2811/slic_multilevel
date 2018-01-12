import numpy as np
import pycuda.driver as pcd
from toolbox import Lung_Cluster as lc

if __name__ == "__main__":
	label_exam = np.load("detection_vision/slic_labels_filtered.npy")
	clcenters1, cllabels1 = lc.cluster_centers_fast(label_exam)
	print clcenters1
	clcenters2, cllabels2 = lc.cluster_centers_gpu(label_exam)
	print clcenters2