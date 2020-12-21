#
# Created by Yue Jiang in June 2020
#


SHOW_BACKGROUND = False       # whether we want to add image background
VISUALIZATION_TYPE = "mesh"   # use "mesh" or "points" to visualize velocity


import math
import trimesh
import pyrender
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm 
from pyrender.constants import RenderFlags
from lib.models.smpl import get_smpl_faces
from lib.utils.vis import *
from lib.models.spin import *
from psbody.mesh.visibility import visibility_compute


# perspective camera for rendering 
class WeakPerspectiveCamera(pyrender.Camera):

    # initialize perspective camera
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

    # get projection matrix for renderer
    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


# velocity renderer for vertex velocity visualization
class VelocityRenderer: 

    # initialize velocity renderer
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False):
 
        # set renderer
        self.resolution = resolution
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=3.0
        )
        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], \
                                            ambient_light=(0.3, 0.3, 0.3))

    def get_visibility(self, verts, cam_dir): 

        # construct the trimesh
        triangle_mesh = trimesh.Trimesh(vertices=verts, \
                                        faces=self.faces, process=False)

        # get the vertices and faces of the trimesh
        vertices = triangle_mesh.vertices
        faces = triangle_mesh.faces

        # compute visibility of vertices
        vis, _ = visibility_compute(v=vertices, f=faces.astype(np.uint32), \
                                    cams=np.double(cam_dir.reshape((1, 3))))
        vertex_visibility = vis[0]

        return vertex_visibility

    # render vertex velocity 
    def render(self, img, verts, cam_transformation, cam_dir, angle=None, \
                axis=None, mesh_filename=None, velocity_colors=None):

        # construct the trimesh
        triangle_mesh = trimesh.Trimesh(vertices=verts, \
                                        faces=self.faces, process=False)

        # transform the trimesh
        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        triangle_mesh.apply_transform(Rx)
        if mesh_filename is not None:
            trimesh.export(mesh_filename)
        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            triangle_mesh.apply_transform(R)

        # get the vertices and faces of the trimesh
        vertices = triangle_mesh.vertices
        faces = triangle_mesh.faces

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
        vis, _ = visibility_compute(v=vertices, f=faces.astype(np.uint32), \
                                    cams=np.double(cam_dir.reshape((1, 3))))
        visibility = np.repeat(np.expand_dims(vis[0], axis=1), 3, axis=1)
        num_vis = np.sum(vis[0])

        # render vertices with visibility
        visibility_colors = np.zeros_like(velocity_colors)
        visibility_colors[vis[0] == 1] = np.array([0,1,0])
        visibility_colors[vis[0] == 0] = np.array([0,0,1])
        triangle_mesh.visual.vertex_colors = visibility_colors
        mesh = pyrender.Mesh.from_trimesh(triangle_mesh)
        # mesh = pyrender.Mesh.from_points(vertices, colors=visibility_colors)

        mesh_node = self.scene.add(mesh, 'mesh') 
        rgb, depth = self.renderer.render(self.scene, flags=render_flags)

        # blend original background and visualized vertices
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask
        if SHOW_BACKGROUND:
            output_img += (1 - valid_mask) * img
        visibility_image = output_img.astype(np.uint8)   
        self.scene.remove_node(mesh_node)

        # create 3D model for rendering velocity
        if VISUALIZATION_TYPE == "points":
            vertices = vertices[visibility == 1].reshape((num_vis, 3))
            velocity_colors = velocity_colors[visibility \
                             == 1].reshape((num_vis, 3))
            mesh = pyrender.Mesh.from_points(vertices, colors=velocity_colors)
        elif VISUALIZATION_TYPE == "mesh":
            triangle_mesh.visual.vertex_colors = velocity_colors
            mesh = pyrender.Mesh.from_trimesh(triangle_mesh)
        else:
            exit("ERROR: VISUALIZATION_TYPE can only be 'points' or 'mesh'")

        # render human with velocity
        mesh_node = self.scene.add(mesh, 'mesh') 
        rgb, depth = self.renderer.render(self.scene, flags=render_flags)

        # blend original background and visualized vertices
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis] 
        output_img = rgb[:, :, :-1] * valid_mask
        if SHOW_BACKGROUND:
            output_img += (1 - valid_mask) * img
        velocity_image = output_img.astype(np.uint8)   
        self.scene.remove_node(mesh_node)

        # define coolwarm mapping
        cmap = plt.get_cmap("coolwarm")
        coolwarm_mapping = matplotlib.cm.ScalarMappable(cmap=cmap)

        # create an example mesh to show green to red
        triangle_mesh_example = triangle_mesh.copy()
        velocity_colors_example = np.zeros_like(velocity_colors)
        max_y = np.max(vertices[:, 1])
        min_y = np.min(vertices[:, 1])
        color_values = (vertices[:, 1] - min_y) / (max_y - min_y)
        velocity_colors_example = coolwarm_mapping.to_rgba(color_values)[:, :-1]
        triangle_mesh_example.visual.vertex_colors = velocity_colors_example 
        mesh_example = pyrender.Mesh.from_trimesh(triangle_mesh_example)
        mesh_node_example = self.scene.add(mesh_example, 'mesh') 
        rgb_example, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb_example[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb_example[:, :, :-1] * valid_mask
        example_image = output_img.astype(np.uint8)  
        self.scene.remove_node(mesh_node_example)
        self.scene.remove_node(cam_node)

        return visibility_image, velocity_image, \
                    example_image
