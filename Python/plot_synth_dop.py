import os
import numpy as np
import matplotlib
from helper import get_spectograms, root_mean_squared_error, color_scale, rolling_window_combine
from tensorflow.keras.models import load_model
import pickle
import cv2
import argparse


def main(args):

	model_path = args.model_path

	lb = pickle.loads(open(model_path+"classifier_classes.lbl", "rb").read())
	autoencoder = load_model(model_path+"autoencoder_weights.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})
	scale_vals = np.load(model_path+"scale_vals.npy")
	fps = 24
	TIME_CHUNK = 3
	max_dopVal = scale_vals[0]
	max_synth_dopVal =  scale_vals[1]
	min_dopVal = scale_vals[2]
	min_synth_dopVal = scale_vals[3]

	vid_f = args.input_video
	in_folder = os.path.dirname(vid_f)
	vid_file_name = args.input_video.split("/")[-1].split(".")[0]

	cap = cv2.VideoCapture(vid_f)

	out_vid = cv2.VideoWriter(in_folder+'/'+vid_file_name+'_output_signal.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, (2425,300))

	synth_doppler_dat_f = in_folder + "/output/" + vid_file_name + "/synth_doppler.npy"
	synth_doppler_dat = np.load(synth_doppler_dat_f)
	synth_spec_pred = get_spectograms(synth_doppler_dat, TIME_CHUNK, fps, synthetic=True, zero_pad=True)
	synth_spec_pred = synth_spec_pred.astype("float32")
	synth_spec_test = (synth_spec_pred - min_synth_dopVal)/(max_synth_dopVal - min_synth_dopVal)
	dop_spec_test = np.zeros_like(synth_spec_test)

	if args.doppler_gt:
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
		original_synth = color_scale(synth_spec_test[idx],matplotlib.colors.Normalize(vmin=0, vmax=np.max(synth_spec_test)),"Initial Synthetic Doppler")
		original_dop = color_scale(dop_spec_test[idx],matplotlib.colors.Normalize(vmin=0, vmax=np.max(dop_spec_test)),"Real World Doppler")
		recon = color_scale(decoded[idx],matplotlib.colors.Normalize(vmin=0, vmax=np.max(decoded)),"Final Synthetic Doppler")
		in_frame = color_scale(frame,None,"Input Video")
		output = np.hstack([in_frame,original_dop, original_synth, recon])
		out_vid.write(output)

	cap.release()
	out_vid.release()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file')

	parser.add_argument('--model_path', type=str, help='Path to DL models')

	parser.add_argument('--doppler_gt', help='Doppler Ground Truth is available for reference', action='store_true')

	args = parser.parse_args()

	main(args)
