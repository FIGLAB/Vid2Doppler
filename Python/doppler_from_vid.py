import argparse
import os
import random
import math
import time
import numpy as np
import shutil

def main(args):

	folder_path = os.path.dirname(os.path.abspath(args.input_video))
	os.system("python run_VIBE.py --input_video %s --output_folder %s" % (args.input_video, folder_path))

	out_path = folder_path + "/output/"

	os.system("python compute_position.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	os.system("python interpolate_frames.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	os.system("python compute_velocity.py --input_video %s --output_folder %s" % (args.input_video, out_path))
	if args.visualize_mesh:
		os.system("python compute_visualization.py --input_video %s --output_folder %s --wireframe" % (args.input_video, out_path))
	os.system("python compute_synth_doppler.py --input_video %s --output_folder %s" % (args.input_video, out_path))

	# free all temporary memory
	image_folder = str(np.load(folder_path + "/image_folder.npy"))
	shutil.rmtree(image_folder)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_video', type=str, help='Input video file')

	parser.add_argument('--visualize_mesh', help='Render visibility mesh and velocity map', action='store_true')

	args = parser.parse_args()

	main(args)
