import numpy as np
import os
from os import listdir
from os.path import isfile, join
from scipy.ndimage import gaussian_filter1d
import argparse
import cv2


N_BINS = 32
DISCARD_BINS = [14,15,16]
GAUSSIAN_BLUR = True
GAUSSIAN_KERNEL = 5
TIME_CHUNK = 1 # 1 second for creating the spectogram


def main(args):

    # get video infomation
    video_file = os.path.basename(args.input_video).replace('.mp4', '')
    out_path = args.output_folder + '/' + video_file + '/'
    video = cv2.VideoCapture("./" + args.input_video)
    fps = video.get(cv2.CAP_PROP_FPS)

    # get frame infomation
    num_frames = len([name for name in \
            os.listdir(out_path + "/frame_velocity") \
            if "frame_" in name])
    output_path = os.path.join(args.output_folder, os.path.basename(\
                                video_file).replace('.mp4', ''))
    if os.path.isfile(output_path + \
                "/../../frames_new.npy"):
        frames = np.load(output_path + \
                    "/../../frames_new.npy", allow_pickle=True)
    else:
        frames = np.load(output_path + \
                    "/../../frames.npy", allow_pickle=True)
    print("frames: ", num_frames)

    # compute synthetic doppler data
    synth_doppler_dat = []
    for frame_idx in frames:
        gen_doppler = np.genfromtxt(out_path + "/frame_velocity/frame_%06d.csv" % frame_idx, delimiter=',')
        velocity = gen_doppler[gen_doppler[:, 1]==1, 0]
        hist = np.histogram(velocity, bins=np.linspace(-2, 2, num=N_BINS+1))[0]
        for bin_idx in DISCARD_BINS:
            hist[bin_idx] = 0
        synth_doppler_dat.append(hist/gen_doppler.shape[0])

    synth_doppler_dat = np.array(synth_doppler_dat)

    if GAUSSIAN_BLUR:
        for i in range(len(synth_doppler_dat)):
            synth_doppler_dat[i] = gaussian_filter1d(synth_doppler_dat[i], GAUSSIAN_KERNEL)

    np.save(out_path+"/synth_doppler.npy",synth_doppler_dat)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_video', type=str, help='input video file')

    parser.add_argument('--output_folder', type=str, help='output folder to write results')

    args = parser.parse_args()

    main(args)
