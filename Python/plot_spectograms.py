import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np 
import os
from os import listdir
from os.path import isfile, join
import argparse
import cv2
from scipy.optimize import curve_fit
from helper import f1, calc_err, get_spectograms, find_scale_vals
import sys

plt.ioff()
plt.style.use('ggplot')

N_BINS = 32
DISCARD_BINS = []
TIME_CHUNK = 3 # 1 second for creating the spectogram

def main(args):

	video_file = os.path.basename(args.input_video).replace('.mp4', '')

	out_path = args.output_folder + '/' + video_file + '/'

	doppler_dat_pos = np.load(args.real_dop)
	synth_doppler_dat = np.load(out_path+"/synth_doppler.npy")

	video = cv2.VideoCapture(out_path + video_file + "_result.mp4")

	plot_path = out_path + "/plots/"
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)

	fps = video.get(cv2.CAP_PROP_FPS)

	for bin_idx in DISCARD_BINS:
		doppler_dat_pos[:,bin_idx] = 0
		synth_doppler_dat[:,bin_idx] = 0

	s_vals = find_scale_vals(synth_doppler_dat, doppler_dat_pos, DISCARD_BINS)
	synth_doppler_dat = s_vals*synth_doppler_dat
	print("Scale values: ", s_vals)
	print("Error is ", calc_err(synth_doppler_dat,doppler_dat_pos))

	dop_spec = get_spectograms(doppler_dat_pos, TIME_CHUNK, fps,synthetic=False,zero_pad=True)
	synth_spec = get_spectograms(synth_doppler_dat, TIME_CHUNK, fps,synthetic=True,zero_pad=True)

	print(synth_spec.shape,synth_doppler_dat.shape)

	y_max = max(np.max(doppler_dat_pos),np.max(synth_doppler_dat))
	norm = matplotlib.colors.Normalize(vmin=0, vmax=y_max)

	for i in range(len(synth_doppler_dat)):
		print(i,len(synth_doppler_dat))
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(30, 6))
		ax1.set_ylim(0, y_max)
		ax2.set_ylim(0, y_max)
		ax1.set_title('Dop.')
		ax2.set_title('Syn.')
		ax1.bar(range(len(doppler_dat_pos[i])),doppler_dat_pos[i])
		ax2.bar(range(len(synth_doppler_dat[i])),synth_doppler_dat[i])
		ax1.set_xticklabels([0,-2,-1.3,-0.7,0,0.7,1.3,2])
		ax2.set_xticklabels([0,-2,-1.3,-0.7,0,0.7,1.3,2])
		dop_xdat =  np.array(range(len(doppler_dat_pos[i])))
		dop_synth_xdat =  np.array(range(len(synth_doppler_dat[i])))
		popt_dop, pcov_dop = curve_fit(f1, dop_xdat, doppler_dat_pos[i])
		popt_synth_dop, pcov_synth_dop = curve_fit(f1, dop_synth_xdat, synth_doppler_dat[i])
		ax1.plot(range(len(doppler_dat_pos[i])), f1(dop_xdat, *popt_dop), '--', color ='blue')
		ax2.plot(range(len(synth_doppler_dat[i])), f1(dop_synth_xdat, *popt_synth_dop), '--', color ='blue')
		ax3.set_title('Dop. Spec')
		ax4.set_title('Syn. Spec')
		ax3.imshow(cm.Greys(norm(dop_spec[i]),bytes=True))
		ax4.imshow(cm.Greys(norm(synth_spec[i]),bytes=True))
		ax3.set_yticklabels([0,-2,-1.3,-0.7,0,0.7,1.3,2])
		ax4.set_yticklabels([0,-2,-1.3,-0.7,0,0.7,1.3,2])
		plt.savefig(plot_path+str(i)+".png")
		plt.close(fig)

	ret, frame_in  = video.read()
	n_frames_total = video.get(cv2.CAP_PROP_FRAME_COUNT)

	video = cv2.VideoCapture(out_path + video_file + "_result.mp4")
	vid_out = cv2.VideoWriter(out_path + video_file + "_spec.mp4",cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_in.shape[1],frame_in.shape[0]*2))
	for i in range(len(synth_doppler_dat)):
		ret, frame  = video.read()
		print(i, frame.shape)
		blankImg = np.zeros(frame_in.shape, np.uint8)
		plt_img = cv2.imread(plot_path+str(i)+".png")
		plt_img = cv2.resize(plt_img,(frame_in.shape[1],frame_in.shape[0]))
		yoff = round((frame_in.shape[0]-plt_img.shape[0])/2)
		xoff = round((frame_in.shape[1]-plt_img.shape[1])/2)
		blankImg[yoff:yoff+plt_img.shape[0], xoff:xoff+plt_img.shape[1]] = plt_img
		res_out = np.vstack((frame,blankImg))
		vid_out.write(res_out)
	vid_out.release()

	os.system("rm -rf %s" % (plot_path))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='input video file')

	parser.add_argument('--real_dop', type=str, help='input real ground-truth doppler file')

	parser.add_argument('--output_folder', type=str, help='output folder to write results')

	args = parser.parse_args()

	main(args)