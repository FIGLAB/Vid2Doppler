#
# Created by Yue Jiang in June 2020
#

 
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import csv
import argparse
import numpy as np
from velocity_renderer import VelocityRenderer
import shutil

def main(args):

    # get input video
    video_file = args.input_video

    # save hand info
    save_hand_csv = args.save_hand_csv

    # define output path
    output_path = os.path.join(args.output_folder, \
                os.path.basename(video_file).replace('.mp4', ''))
    csv_folder_path = os.path.join(output_path, "frame_position/")
    if save_hand_csv:
        hand_csv_folder_path = os.path.join(output_path, "hand_frame_position/")

    output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(csv_folder_path, exist_ok=True)
    if save_hand_csv:
        os.makedirs(hand_csv_folder_path, exist_ok=True)

    # get defined parameters
    image_folder = str(np.load(output_path + "/../../image_folder.npy"))
    orig_width = np.load(output_path + "/../../orig_width.npy", \
                                            allow_pickle=True)
    orig_height = np.load(output_path + "/../../orig_height.npy", \
                                            allow_pickle=True)

    # get frame results
    frame_results = np.load(output_path + \
                    "/../../frame_results.npy", allow_pickle=True)
    frames = np.load(output_path + \
                    "/../../frames.npy", allow_pickle=True)

    # define a renderer
    renderer = VelocityRenderer(resolution=(orig_width, \
        orig_height), orig_img=True, wireframe=args.wireframe)
    
    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    # define camera origin position
    camera_orig = [float(i) for i in args.camera_orig[1:-1].split(',')] 

    # render each frame
    index = 0
    for frame_idx in frames:

        # loop over each frame
        for person_id, person_data in frame_results[index].items():
            frame_verts = person_data['verts']

            # get camera direction
            camera_dir = camera_orig / np.linalg.norm(camera_orig)

            # render images
            vertex_visibility = renderer.get_visibility(frame_verts, camera_dir)

        # save each vertex position, velocity and visibility
        save_body_csv = True 
        if save_body_csv: 
            with open(csv_folder_path + "frame_%06d.csv" \
                        % frame_idx, mode='w') as frame_info:
                frame_info_writer = csv.writer(frame_info, \
                    delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for i in range(len(frame_verts)):
                    frame_info_writer.writerow([str(frame_verts[i][0]), \
                        str(frame_verts[i][1]), str(frame_verts[i][2]), \
                                        str(vertex_visibility[i])])
        if save_hand_csv:
             with open(hand_csv_folder_path + "frame_%06d.csv" \
                                % frame_idx, mode='w') as frame_info:
                frame_info_writer = csv.writer(frame_info, \
                    delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for i in range(2000, 2500):
                    frame_info_writer.writerow([str(frame_verts[i][0]), \
                             str(frame_verts[i][1]), str(frame_verts[i][2]), \
                                str(vertex_visibility[i])])
        index += 1


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