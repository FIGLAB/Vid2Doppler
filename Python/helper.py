import numpy as np
import cv2
import sys

def get_spectograms(dop_dat, t_chunk, frames_per_sec, t_chunk_overlap=None, synthetic=False,zero_pad=False):
	frame_overlap = 1
	if t_chunk_overlap is not None:
		frame_overlap = int(t_chunk_overlap * frames_per_sec)
	frame_chunk = int(t_chunk * frames_per_sec)
	if zero_pad == True:
		zero_padding = np.zeros((32,frame_chunk-1))
		dop_dat_spec = np.hstack((zero_padding,np.transpose(dop_dat)))
	else:
		dop_dat_spec = np.transpose(dop_dat)
	spectogram = []
	if zero_pad == True:
		for i in range(0,len(dop_dat), frame_overlap):
			spec = dop_dat_spec[:,i:i+frame_chunk]
			if synthetic == True:
					spec = cv2.GaussianBlur(spec,(5,5),0)
			spectogram.append(spec)
	else:
		for i in range(0,len(dop_dat)-frame_chunk, frame_overlap):
			spec = dop_dat_spec[:,i:i+frame_chunk]
			if synthetic == True:
				if zero_pad == True:
					spec = cv2.GaussianBlur(spec,(5,5),0)
			spectogram.append(spec)
	spectogram = np.array(spectogram)
	return spectogram
