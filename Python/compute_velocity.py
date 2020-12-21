#
# Created by Yue Jiang in June 2020
#

import argparse
import csv
import math
import cv2
import os
import numpy as np 


def main(args):

    # define camera origin position
    camera_orig = [float(i) for i in args.camera_orig[1:-1].split(',')] 

    # get video file name
    video_file = args.input_video.split("/")[-1].split(".")[0]

    # get fps of the video
    video = cv2.VideoCapture(args.input_video)
    fps = video.get(cv2.CAP_PROP_FPS)

    # save hand info
    save_hand_csv = args.save_hand_csv

    # set output flles
    output_path = os.path.join(args.output_folder, os.path.basename(\
                                video_file).replace('.mp4', ''))
    csv_folder_path = os.path.join(output_path, "frame_velocity/")
    os.makedirs(csv_folder_path, exist_ok=True)
    if save_hand_csv:
        hand_csv_folder_path = os.path.join(output_path, "hand_frame_velocity/")  
        os.makedirs(hand_csv_folder_path, exist_ok=True)

    # get the number of frames
    num_frames = len([name for name in os.listdir(args.output_folder \
                            + video_file + "/frame_position") \
                            if "frame_" in  name])

    # read frame info as numpy arrays from csv files
    vertex_position = []
    vertex_visibilty = []
    if save_hand_csv:
        hand_vertex_position = []
        hand_vertex_visibilty = []
    if os.path.isfile(output_path + \
                "/../../frames_new.npy"):
        frames = np.load(output_path + \
                    "/../../frames_new.npy", allow_pickle=True)
    else:
        frames = np.load(output_path + \
                    "/../../frames.npy", allow_pickle=True)

    for frame_idx in frames:

        # read frame info for human body
        frame_info = np.genfromtxt(args.output_folder + video_file \
            + "/frame_position/frame_%06d.csv" \
                            % frame_idx, delimiter=',')
        vertex_position.append(frame_info[:, :3])
        vertex_visibilty.append(frame_info[:, 3:])

        # read frame info for human hand
        if save_hand_csv:
            hand_frame_info = np.genfromtxt(args.output_folder + video_file \
                + "/hand_frame_position/frame_%06d.csv" \
                                        % frame_idx, delimiter=',')
            hand_vertex_position.append(hand_frame_info[:, :3])
            hand_vertex_visibilty.append(hand_frame_info[:, 3:])

    # change position and visibility lists to numpy arrays
    vertex_position = np.array(vertex_position)
    vertex_visibilty = np.array(vertex_visibilty)
    if save_hand_csv:
        hand_vertex_position = np.array(hand_vertex_position)
        hand_vertex_visibilty = np.array(hand_vertex_visibilty)

    # compute radial velocity for human body and hand
    vertex_velocity_list = []
    if save_hand_csv:
        hand_vertex_velocity_list = []
 
    for frame_idx in range(len(frames)):

        # skip the first frame
        if frame_idx < 1:
            vertex_velocity = np.expand_dims(np.zeros_like(\
                        vertex_position[frame_idx][:,0]), axis=1)
            vertex_velocity_list.append(vertex_velocity)
            if save_hand_csv:
                hand_vertex_velocity = np.expand_dims(np.zeros_like(\
                            hand_vertex_position[frame_idx][:,0]), axis=1)
                hand_vertex_velocity_list.append(hand_vertex_velocity)

        # Calculate radial velocity
        else:

            # compute radial velocity for human body
            v = vertex_position[frame_idx] - vertex_position[frame_idx-1]
            p_t_1 = vertex_position[frame_idx-1] - camera_orig
            p_t_2 = vertex_position[frame_idx] - camera_orig
            v = p_t_2 - p_t_1
            dot_prod = np.multiply(v, p_t_2).sum(axis=1)
            mag = np.linalg.norm(p_t_2, axis=1)
            vertex_velocity = np.expand_dims(-(dot_prod / mag) * fps, axis=1)
            vertex_velocity_list.append(vertex_velocity)

            # compute radial velocity for the person's left hand
            if save_hand_csv:
                v = hand_vertex_position[frame_idx] \
                        - hand_vertex_position[frame_idx-1]
                p_t_1 = hand_vertex_position[frame_idx-1] - camera_orig
                p_t_2 = hand_vertex_position[frame_idx] - camera_orig
                v = p_t_2 - p_t_1
                dot_prod = np.multiply(v, p_t_2).sum(axis=1)
                mag = np.linalg.norm(p_t_2, axis=1)
                hand_vertex_velocity = np.expand_dims(-(dot_prod \
                                            / mag) * fps, axis=1)
                hand_vertex_velocity_list.append(hand_vertex_velocity)

    # compute velocity mean for human body using convolution
    velocity_map = np.array(vertex_velocity_list)
    velocity_map = velocity_map[:,:,0]
    for j in range(velocity_map.shape[1]):
       velocity_map[:,j] = np.convolve(velocity_map[:,j], \
                                np.ones((5,))/5, mode='same')
    velocity_map = np.expand_dims(velocity_map, axis=2)

    # compute velocity mean for human hand using convolution
    if save_hand_csv:
        hand_velocity_map = np.array(hand_vertex_velocity_list)
        hand_velocity_map = hand_velocity_map[:,:,0]
        for j in range(hand_velocity_map.shape[1]):
           hand_velocity_map[:,j] = np.convolve(hand_velocity_map[:,j], \
                                        np.ones((5,))/5, mode='same')
        hand_velocity_map = np.expand_dims(hand_velocity_map, axis=2)

    # save velocities and visibilities
    index = 0
    for frame_idx in frames:

        # concatenate velocities and visibilities
        frame_info = np.concatenate((velocity_map[index], \
                                vertex_visibilty[index]), axis=1)
        if save_hand_csv:
            hand_frame_info = np.concatenate((hand_velocity_map[index], \
                                hand_vertex_visibilty[index]), axis=1)

        # save each vertex velocity and visibility
        np.savetxt(csv_folder_path + "frame_%06d.csv" % frame_idx, \
                                            frame_info, delimiter=",")
        if save_hand_csv:
            np.savetxt(hand_csv_folder_path + "frame_%06d.csv" \
                        % frame_idx, hand_frame_info, delimiter=",")
        index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_video', type=str,
                        help='input video file')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--camera_orig', type=str, default="[0,0,-10]",
                        help='camera origin position')

    parser.add_argument('--save_hand_csv', action='store_true',
                        help='render all meshes as wireframes.')

    args = parser.parse_args()

    main(args)




    