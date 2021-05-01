import os
import numpy as np
import matplotlib
from helper import get_spectograms, root_mean_squared_error, color_scale, rolling_window_combine
from tensorflow.keras.models import load_model
import pickle
import cv2

dop_gt = True

path = "../"
data_path = path + "data/"
model_path = path + "models/"

lb = pickle.loads(open(model_path+"classifier_classes.lbl", "rb").read())
autoencoder = load_model(model_path+"autoencoder_weights.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})
scale_vals = np.load(model_path+"scale_vals.npy")
fps = 24
TIME_CHUNK = 3
max_dopVal = scale_vals[0]
max_synth_dopVal =  scale_vals[1]
min_dopVal = scale_vals[2]
min_synth_dopVal = scale_vals[3]


in_folder = "../video/"
vid_f = in_folder + "/sample_video.mp4"
cap = cv2.VideoCapture(vid_f)

out_vid = cv2.VideoWriter(in_folder+'/output_signal.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, (2425,300))

synth_doppler_dat_f = in_folder + "/output/sample_video/synth_doppler.npy"
synth_doppler_dat = np.load(synth_doppler_dat_f)
synth_spec_pred = get_spectograms(synth_doppler_dat, TIME_CHUNK, fps, synthetic=True, zero_pad=True)
synth_spec_pred = synth_spec_pred.astype("float32")
synth_spec_test = (synth_spec_pred - min_synth_dopVal)/(max_synth_dopVal - min_synth_dopVal)
dop_spec_test = np.zeros_like(synth_spec_test)

if dop_gt:
	doppler_dat_pos_f = in_folder + "/doppler_gt.npy"
	doppler_dat_pos = np.load(doppler_dat_pos_f)
	dop_spec = get_spectograms(doppler_dat_pos, TIME_CHUNK, fps, zero_pad=True)
	dop_spec = dop_spec.astype("float32")
	dop_spec_test = (dop_spec - min_dopVal)/(max_dopVal - min_dopVal)

decoded = autoencoder.predict(synth_spec_test)
decoded = decoded[:,:,:,0]

decoded = rolling_window_combine(decoded)

y_max = max(np.max(decoded),np.max(synth_spec_test),np.max(dop_spec_test))
norm = matplotlib.colors.Normalize(vmin=0, vmax=y_max)

for idx in range(0, len(dop_spec_test)):
	ret, frame = cap.read()
	original_synth = color_scale(synth_spec_test[idx],norm,"Initial Synthetic Doppler")
	original_dop = color_scale(dop_spec_test[idx],norm,"Real World Doppler")
	recon = color_scale(decoded[idx],norm,"Final Synthetic Doppler")
	in_frame = color_scale(frame,norm,"Input Video")
	output = np.hstack([in_frame,original_dop, original_synth, recon])
	out_vid.write(output)
	# cv2.imshow("output",output)
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break

cap.release()
out_vid.release()
