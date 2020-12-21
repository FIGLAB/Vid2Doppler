# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import math
import trimesh
import pyrender
import numpy as np
import torch
from pyrender.constants import RenderFlags
from lib.models.smpl import get_smpl_faces
from lib.utils.vis import *
from lib.models.spin import *
from psbody.mesh.visibility import visibility_compute


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=3.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)


    def render(self, img, verts, cam_transformation, cam_dir, angle=None, axis=None, mesh_filename=None, velocity_colors=None):

        # construct the mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        # transform the mesh
        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)
        if mesh_filename is not None:
            mesh.export(mesh_filename)
        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        # get the vertices and faces of the mesh
        vertices = mesh.vertices
        faces = mesh.faces

        # set camera pose
        sx, sy, tx, ty = cam_transformation
        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000. 
        )
        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        # set rendering flags
        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        # compute visibility of vertices
        vis, _ = visibility_compute(v=vertices, f=faces.astype(np.uint32), cams=np.double(np.array([[0,0,1]])))
        visibility = np.repeat(np.expand_dims(vis[0], axis=1), 3, axis=1)
        num_vis = np.sum(vis[0])

        # render vertices with visibility
        visibility_colors = np.zeros_like(velocity_colors)
        visibility_colors[vis[0] == 1] = np.array([0,1,0])
        visibility_colors[vis[0] == 0] = np.array([0,0,1])
        mesh = pyrender.Mesh.from_points(vertices, colors=visibility_colors)
        mesh_node = self.scene.add(mesh, 'mesh') 
        rgb, depth = self.renderer.render(self.scene, flags=render_flags)

        # blend original background and visualized vertices
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img * 0
        visibility_image = output_img.astype(np.uint8)   
        self.scene.remove_node(mesh_node)

        # render vertices with velocity
        vertices = vertices[visibility == 1].reshape((num_vis, 3))
        velocity_colors = velocity_colors[visibility == 1].reshape((num_vis, 3))
        mesh = pyrender.Mesh.from_points(vertices, colors=velocity_colors)
        mesh_node = self.scene.add(mesh, 'mesh') 
        rgb, depth = self.renderer.render(self.scene, flags=render_flags)

        # blend original background and visualized vertices
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img * 0
        velocity_image = output_img.astype(np.uint8)   
        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return visibility_image, velocity_image
