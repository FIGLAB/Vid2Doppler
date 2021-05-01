import numpy as np
import cv2
import sys
import imutils
from matplotlib import cm

def rolling_window_combine(X_in):
	for i in range(1,len(X_in)):
		X_in[i][:,:-1] = X_in[i-1][:,1:]
	return X_in

def color_scale(img, norm,text=None):
	if len(img.shape) == 2:
		img = cm.magma(norm(img),bytes=True)
	img = imutils.resize(img, height=300)
	if img.shape[2] == 4:
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
	if text is not None:
		img = cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
	return img

def rolling_average(X_in):
	for i in range(1,X_in.shape[0]-1):
		X_in[i] = (X_in[i] + X_in[i+1] + X_in[i-1])/3
	return X_in

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square((y_pred*255) - (y_true*255))))

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
