import argparse
import os
import random
import math
import time
import numpy as np
import shutil

def main(args):
	
	camera_origins = ["[0,0,10]"] 
	
	if args.synth_origins > 1:
		height_vals = ['-0.5','0','0.5']
		for y_val in height_vals:
			thetas = random.sample(range(0, 180), args.synth_origins)
			for theta in thetas:
				x_val = int(10*math.cos(math.radians(theta)))
				z_val = int(10*math.sin(math.radians(theta)))
				c_orig = "["+str(x_val)+","+str(y_val)+","+str(z_val)+"]"
				camera_origins.append(c_orig)

	folder_path = os.path.dirname(os.path.abspath(args.input_video))
	os.system("python run_VIBE.py --input_video %s --output_folder %s" % (args.input_video, folder_path))

	for camera_orig in camera_origins:
		print("Camera Orig:", camera_orig)

		camera_orig_str = [str(i) for i in camera_orig[1:-1].split(',')] 
		out_path = folder_path + "/output_" + camera_orig_str[0] + '_' + camera_orig_str[1] + '_' + camera_orig_str[2] +'/'

		os.system("python compute_position.py --input_video %s --output_folder %s --camera_orig %s" % (args.input_video, out_path, camera_orig))
		os.system("python interpolate_frames.py --input_video %s --output_folder %s --camera_orig %s" % (args.input_video, out_path, camera_orig))
		os.system("python compute_velocity.py --input_video %s --output_folder %s --camera_orig %s" % (args.input_video, out_path, camera_orig))
		if args.visualize_mesh: 
			if args.wireframe:
				os.system("python compute_visualization.py --input_video %s --output_folder %s --camera_orig %s --wireframe" % (args.input_video, out_path, camera_orig))
			else:
				os.system("python compute_visualization.py --input_video %s --output_folder %s --camera_orig %s" % (args.input_video, out_path, camera_orig))
		os.system("python compute_synth_doppler.py --input_video %s --output_folder %s" % (args.input_video, out_path))
		if args.plot_spectogram:
			if args.visualize_mesh == False: # Mesh rendering is a pre-requisite of this
					os.system("python compute_visualization.py --input_video %s --output_folder %s --camera_orig %s" % (args.input_video, out_path, camera_orig))
			os.system("python plot_spectograms.py --real_dop %s --input_video %s --output_folder %s" % (folder_path+"/doppler_gt.npy", args.input_video, out_path))

	# free all temporary memory
	image_folder = str(np.load(folder_path + "/image_folder.npy"))
	shutil.rmtree(image_folder)

	
def check_positive(value):
	ivalue = int(value)
	if ivalue <= 0:
		raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
	return ivalue


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file')

	parser.add_argument('--visualize_mesh', help='Render visibility mesh and velocity map', action='store_true')

	parser.add_argument('--wireframe', help='Wire frame rendering', action='store_true')

	parser.add_argument('--plot_spectogram', help='Plot side-by-side spectogram with ground truth doppler is available', action='store_true')

	parser.add_argument('--synth_origins', type=check_positive, default="1", help='positive number of synthetic doppler origins at each level')

	args = parser.parse_args()

	main(args)