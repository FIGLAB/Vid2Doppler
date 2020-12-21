#
# Created by Yue Jiang in August 2020
#

import argparse
import os
import numpy as np 
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import csv


def main(args):

    # get video file name
    video_file = args.input_video.split("/")[-1].split(".")[0]

    # save hand info
    save_hand_csv = args.save_hand_csv

    # define output path
    output_path = os.path.join(args.output_folder, \
                os.path.basename(video_file).replace('.mp4', ''))
    
    # get frames
    frames = np.load(output_path + \
                    "/../../frames.npy", allow_pickle=True)
    start_frame = frames[0]
    end_frame = frames[-1]

    # get camera transformation 
    orig_cameras = np.genfromtxt(args.output_folder + video_file \
                + "/../../orig_cam.csv", delimiter=',')
    new_cameras = []

    # interpolate frames
    np.save(output_path + "/../../frames_new.npy", np.arange(start_frame, end_frame + 1))
    for i in range(len(frames) - 1):
        new_cameras.append(orig_cameras[i])
        if frames[i] + 1 != frames[i+1]:
            for f in range(frames[i] + 1, frames[i+1]):

                # get camera tansformation from the previous avaalable frame
                new_cameras.append(orig_cameras[i])

                # read frame info for human body
                previous_frame = np.genfromtxt(args.output_folder + video_file \
                    + "/frame_position/frame_%06d.csv" % frames[i], delimiter=',')
                next_frame = np.genfromtxt(args.output_folder + video_file \
                    + "/frame_position/frame_%06d.csv" % frames[i+1], delimiter=',')
                
                # interpolate to get the current frame
                current_frame = np.zeros_like(previous_frame)
                current_frame[:, 3] = np.maximum(previous_frame[:, 3], \
                                                        next_frame[:, 3])
                current_frame[:, :3] = (previous_frame[:, :3] * (frames[i+1] - f) \
                                        + next_frame[:, :3] * (f - frames[i])) \
                                                    / (frames[i+1] - frames[i])
                # read frame info for human hand
                if save_hand_csv:
                    hand_previous_frame = np.genfromtxt(args.output_folder + video_file \
                        + "/hand_frame_position/frame_%06d.csv" \
                                            % frames[i], delimiter=',')
                    hand_next_frame = np.genfromtxt(args.output_folder + video_file \
                        + "/hand_frame_velocity/frame_%06d.csv" \
                                            % frames[i+1], delimiter=',')

                    # interpolate to get the current frame
                    hand_current_frame = np.zeros_like(hand_previous_frame)
                    hand_current_frame[:, 3] = np.maximum(hand_previous_frame[:, 3], \
                                                            hand_next_frame[:, 3])
                    hand_current_frame[:, :3] = (hand_previous_frame[:, :3] * (frames[i+1] - f) \
                                            + hand_next_frame[:, :3] * (f - frames[i])) \
                                                        / (frames[i+1] - frames[i])

                # save each vertex velocity and visibility
                np.savetxt(args.output_folder + video_file \
                        + "/frame_position/frame_%06d.csv" % f, \
                                            current_frame, delimiter=",")
                if save_hand_csv:
                    np.savetxt(args.output_folder + video_file \
                         + "/hand_frame_velocity/frame_%06d.csv" \
                                % f, hand_current_frame, delimiter=",")

    # update camera transformation
    new_cameras.append(orig_cameras[-1])
    np.savetxt(args.output_folder + video_file \
            + "/../../orig_cam_new.csv", np.array(new_cameras), delimiter=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_video', type=str,
                        help='input video path or youtube link')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--camera_orig', type=str, default="[0,0,-10]",
                        help='camera origin position')

    parser.add_argument('--save_hand_csv', action='store_true',
                        help='render all meshes as wireframes.')

    args = parser.parse_args()

    main(args)