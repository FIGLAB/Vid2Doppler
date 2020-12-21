#
# Created by Yue Jiang in June 2020
#

 
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import csv
import torch
import shutil
import colorsys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm 
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
from lib.models.vibe import VIBE_Demo
from lib.dataset.inference import Inference
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker 
from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)
import shutil


def main(args):

    # check GPU availability
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get input video
    video_file = args.input_video

    # save hand info
    save_hand_csv = args.save_hand_csv

    # ========= [Optional] download the youtube video ========= #
    if video_file.startswith('https://www.youtube.com'):
        print(f'Donwloading YouTube video \"{video_file}\"')
        video_file = download_youtube_clip(video_file, '/tmp')
        if video_file is None:
            exit('Youtube url is not valid!')
        print(f'YouTube Video has been downloaded to {video_file}...')

    # check video existence
    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')

    # set output flles
    output_path = args.output_folder
    image_folder, num_frames, img_shape = video_to_images(video_file, \
                    "/tmp/" + output_path.split("/")[-1], return_info=True)
    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    # get the frame rate (frames per second) of the input video
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)

    # ========= Run tracking ========= #
    bbox_scale = 1.1
  
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
    )
    tracking_results = mot(image_folder)

    # only focus on the frame with the first person
    largest_num_frames = 0
    largest_person = None
    for person_id in list(tracking_results.keys()):
        num_frames = tracking_results[person_id]['frames'].shape[0]
        if num_frames <= largest_num_frames:
            del tracking_results[person_id]
        else:
            largest_num_frames = tracking_results[person_id]['frames'].shape[0]
            if largest_person != None:
                del tracking_results[largest_person]
            largest_person = person_id 

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    print(f'Running VIBE on each tracklet...')
    vibe_results = {}

    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']
        np.save(output_path + "/frames", frames)

        # inference data of each person
        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            scale=bbox_scale,
        )
        bboxes = dataset.bboxes
        frames = dataset.frames
       
        # load data
        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16)

        # extract data
        with torch.no_grad():
            pred_cam, pred_verts, pred_pose = [], [], []
            for batch in dataloader:
                batch = batch.unsqueeze(0)
                batch = batch.to(device)
                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]
                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1)) 
            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            del batch

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()

        # get camera pose
        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        # get result information
        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'bboxes': bboxes,
            'frame_ids': frames,
        }
        np.savetxt(output_path + "/pred_cam.csv", pred_cam, delimiter=",")
        np.savetxt(output_path + "/orig_cam.csv", orig_cam, delimiter=",")
        vibe_results[person_id] = output_dict 
    del model

    frame_results = prepare_rendering_results(vibe_results, len(frames))
    np.save(output_path + "/frame_results", frame_results)
    np.save(output_path + "/image_folder", image_folder)
    np.save(output_path + "/orig_width", orig_width)
    np.save(output_path + "/orig_height", orig_height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_video', type=str,
                        help='input video path or youtube link')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--camera_orig', type=str, default="[0,0,-10]",
                        help='camera origin position')

    parser.add_argument('--save_hand_csv', action='store_true',
                        help='render all meshes as wireframes.')

    args = parser.parse_args()

    main(args)