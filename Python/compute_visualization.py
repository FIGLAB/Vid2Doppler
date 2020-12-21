#
# Created by Yue Jiang in June 2020
#

import argparse
import csv
import math
import cv2
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from velocity_renderer import VelocityRenderer
import colorsys


def main(args):

    # define camera origin position
    camera_orig = [float(i) for i in args.camera_orig[1:-1].split(',')] 

    # get video file name
    video_file = args.input_video.split("/")[-1].split(".")[0]

    # save hand info
    save_hand_csv = args.save_hand_csv

    # get fps of the video
    video = cv2.VideoCapture(args.input_video)    
    fps = video.get(cv2.CAP_PROP_FPS)

    # define video writer
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    if args.wireframe:
        out = cv2.VideoWriter(args.output_folder + video_file + "/" + video_file \
                            + "_result_bottom.mp4", fourcc, fps, \
                            (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))) 
    else:
        out = cv2.VideoWriter(args.output_folder + video_file + "/" + video_file \
                            + "_result_velocity.mp4", fourcc, fps, \
                            (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))) 

    # get the number of frames
    num_frames = len([name for name in \
            os.listdir(args.output_folder + video_file + "/frame_position") \
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
    print("visualized frames: ", len(frames))

    # read frame info as numpy arrays from csv files
    vertex_position = []
    vertex_velocity = []
    if save_hand_csv:
        hand_vertex_position = []
        hand_vertex_velocity = []
    for frame_idx in frames:

        # read frame info for human body
        frame_info = np.genfromtxt(args.output_folder + video_file \
            + "/frame_position/frame_%06d.csv" % frame_idx, delimiter=',')
        vertex_position.append(frame_info[:, :3])
        frame_info = np.genfromtxt(args.output_folder + video_file \
            + "/frame_velocity/frame_%06d.csv" % frame_idx, delimiter=',')
        vertex_velocity.append(frame_info[:, 0])

        # read frame info for human hand
        if save_hand_csv:
            hand_frame_info = np.genfromtxt(args.output_folder + video_file \
                + "/hand_frame_position/frame_%06d.csv" \
                                    % frame_idx, delimiter=',')
            hand_vertex_position.append(hand_frame_info[:, :3])
            hand_frame_info = np.genfromtxt(args.output_folder + video_file \
                + "/hand_frame_velocity/frame_%06d.csv" \
                                    % frame_idx, delimiter=',')
            hand_vertex_velocity.append(hand_frame_info[:, 0])

    # get the number of vertices
    num_vertices = vertex_position[0].shape[0]
    if save_hand_csv:
        num_hand_vertices = hand_vertex_position[0].shape[0]

    # get predicted camera positions from the model
    orig_cameras = np.genfromtxt(args.output_folder + video_file \
                + "/../../orig_cam.csv", delimiter=',')

    # change position and velocity lists to numpy arrays
    vertex_position = np.array(vertex_position)
    vertex_velocity = np.array(vertex_velocity)
    if save_hand_csv:
        hand_vertex_position = np.array(hand_vertex_position)
        hand_vertex_velocity = np.array(hand_vertex_velocity)

    count = 0
    frames = frames[:50]
    for frame_idx in frames:

        # capture frames in the video 
        ret, frame = video.read() 
        print(count)
        count +=1

        # define renderer
        if frame_idx == 0:
            orig_height, orig_width = frame.shape[:2]
            renderer = VelocityRenderer(resolution=(orig_width, \
                                    orig_height), orig_img=True, \
                                    wireframe=args.wireframe)

        # skip frames without the main person
        if frame_idx not in frames:
            continue

        # compute velocity colors 
        velocity_colors = np.zeros((num_vertices, 3))
        max_velocity = 2
        min_velocity = -2 
 
        # define coolwarm mapping
        cmap = plt.get_cmap("RdYlBu")
        norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
        coolwarm_mapping = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)

        # get velocity colors 
        velocity_colors = coolwarm_mapping.to_rgba(\
                                        vertex_velocity[frame_idx])[:, :-1]

        # get camera direction
        camera_dir = camera_orig / np.linalg.norm(camera_orig)

        velocity_colors = np.ones((num_vertices, 3))
        velocity_colors[:,0] *= 0.5
        velocity_colors[:,1] *= 0.6

        count_frame = 1 
        position = vertex_position[frame_idx]
        for i in range(7):
            if frame_idx > i:
                position += vertex_position[frame_idx-i-1]
                count_frame += 1
            if frame_idx < len(frames) - (i + 1):
                position += vertex_position[frame_idx+i+1]
                count_frame += 1
        position /= count_frame

        # render images
        visibility_image, velocity_image, example_image = renderer.render(
            frame, position,
            # vertex_position[frame_idx],
            cam_transformation=orig_cameras[frame_idx],
            cam_dir = camera_dir,
            velocity_colors=velocity_colors,
            mesh_filename=None,
            angle=45,
            axis=[ -0.3826834, 0, 0, 0.9238795 ]
        ) 

        frame = velocity_image#velocity_image * (velocity_image > 0) + frame * (velocity_image == 0)

        # frame = np.concatenate([frame, visibility_image, \
        #                         velocity_image, example_image], axis=1) 

        if args.show_text and save_hand_csv:

            # get the string for position result
            position_result_str = "position: " + str(round(sum(\
                        hand_vertex_position[frame_idx][:,0]) / \
                                num_hand_vertices, 2)) + ", " \
                        + str(round(sum(hand_vertex_position[frame_idx][:,1]) \
                         / num_hand_vertices, 2)) + ", " \
                        + str(round(sum(hand_vertex_position[frame_idx][:,2]) \
                            / num_hand_vertices, 2))

            # get the string for radial velocity result, they don't have the first frame 
            radial_result_str = "radial: " + str(round(sum(\
                hand_vertex_velocity[frame_idx]) / num_hand_vertices, 2))
      
            # describe the type of font to be used
            font = cv2.FONT_HERSHEY_SIMPLEX 
          
            # Use putText() method for inserting text on video 
            cv2.putText(frame,  
                        position_result_str,  
                        (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2, 50),  
                        font, 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4) 
            cv2.putText(frame,  
                        radial_result_str,  
                        (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2, 100),  
                        font, 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4) 

        # Display the resulting frame 
        out.write(frame)
      
    # release the cap object 
    out.release() 

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_video', type=str,
                        help='input video file')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--camera_orig', type=str, default="[0,0,-10]",
                        help='camera origin position')

    parser.add_argument('--show_text', action='store_true', 
                        help='whether we show the text') 

    parser.add_argument('--save_hand_csv', action='store_true',
                        help='render all meshes as wireframes.')

    args = parser.parse_args()

    main(args)