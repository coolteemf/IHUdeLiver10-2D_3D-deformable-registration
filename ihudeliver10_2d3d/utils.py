import os
from typing import Union
from itertools import product as prod

import torch
import nrrd
import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from deepdrr import Volume, geo, Projector
from deepdrr.geo import CameraProjection, CameraIntrinsicTransform, FrameTransform

geoR = geo.FrameTransform.from_rotation
geoT = geo.FrameTransform.from_translation




def crop_from_roi_2D(roi_2D):
    """This is specific to the output of the DeepDRR Projection class.
    The input ROI is the ROI of the projection of the camera grid.
    The return coordinates are in the following order: [x_min, x_max, y_min, y_max].
    It is made to crop the non-transposed output of the project function.
    """
    return [roi_2D[1], roi_2D[3], roi_2D[0], roi_2D[2]]


def crop_roi(roi_to_crop: list, roi_to_crop_from: list, crop_max: bool = True, crop_min: bool = True):
    """
    When cropping a volume, the indexing of the volume changes. This function translates the roi_to_crop
    to the new indexing.
    """
    roi_cropped = [roi_to_crop[0] - roi_to_crop_from[0], roi_to_crop[1] - roi_to_crop_from[1],
                   roi_to_crop[2] - roi_to_crop_from[2], roi_to_crop[3] - roi_to_crop_from[0],
                   roi_to_crop[4] - roi_to_crop_from[1], roi_to_crop[5] - roi_to_crop_from[2]]
    if crop_max:
        roi_cropped = [roi_cropped[0], roi_cropped[1], roi_cropped[2],
                       min(roi_cropped[3], roi_to_crop_from[3] - roi_to_crop_from[0]),
                       min(roi_cropped[4], roi_to_crop_from[4] - roi_to_crop_from[1]),
                       min(roi_cropped[5], roi_to_crop_from[5] - roi_to_crop_from[2])]
    if crop_min:
        roi_cropped = [max(roi_cropped[0], 0), max(roi_cropped[1], 0), max(roi_cropped[2], 0),
                       roi_cropped[3], roi_cropped[4], roi_cropped[5]]

    return roi_cropped


def check_point_in_bb(box: torch.Tensor, point):
    if not tuple(box.shape) == (2,3):
        raise ValueError
    if not (box[1] > box[0]).all():
        raise ValueError
    box_size = box[1] - box[0]
    d = point - box[0]
    return torch.logical_and((d >= 0.).all(dim=-1),
                             (d <= box_size).all(dim=-1))


def project(camera_projection, volume, source_to_detector_distance, gamma=None, neglog=True, invert=True):
    with Projector(volume,
                   neglog=neglog,
                   camera_intrinsics=camera_projection.intrinsic,
                   source_to_detector_distance=source_to_detector_distance) as projector:
        projected = projector(camera_projection)
        if invert:
            projected = 1 - projected
        if gamma is not None:
            projected = projected ** (1 / gamma)
    return projected


def load_volume_data(path: str):
    if path.endswith('nii'):
        if not os.path.exists(path):
            path = path.replace('.nii', '.nii.gz')
        return nib.load(path)
    elif path.endswith('nii.gz'):
        if not os.path.exists(path):
            path = path.replace('.nii.gz', '.nii')
        return nib.load(path)
    elif path.endswith('nrrd'):
        map_directions = {'left': -1, 'right':1, 'posterior': -1, 'anterior':1, 'inferior': -1, 'superior':1}
        volume_data = nrrd.read(path)
        directions = volume_data[1]['space'].split('-')
        if directions[0] != 'left' or directions[0] != 'right':
            raise ValueError("First dim must be left-right")
        if directions[1] != 'posterior' or directions[1] != 'anterior':
            raise ValueError("Second dim must be posterior-anterior")
        if directions[2] != 'inferior' or directions[2] != 'superior':
            raise ValueError("Third dim must be inferior-superior")
        directions = np.array([map_directions[direction] for direction in directions])
        affine = np.eye(4)
        affine[:3,:3] = volume_data[1]['space directions'] * directions
        affine[:3,3] = volume_data[1]['space origin'] * directions
        return nib.Nifti1Image(volume_data[0], affine)
    else:
        raise ValueError("Volume must be in nii or nrrd format")


def Rotation_to_align(vector_initial, vector_final):
    v0 = vector_initial / np.linalg.norm(vector_initial)
    v1 = vector_final / np.linalg.norm(vector_final)
    v = np.cross(v0, v1)
    s = np.linalg.norm(v)
    if s > 1e-6:
        c = np.dot(v0, v1)
        I = np.eye(3)
        vx = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        return I + vx + vx @ vx * (1 - c) / (s ** 2)
    else:
        return np.eye(3)


def R_from_IJK_to_C3D(volume, camera_normal, camera_intrinsic):
    """Compute the orientation of the camera so that its normal is colinear to the given camera_normal.
    camera_normal is understood as an IJK vector and will thus be converted to anatomical."""
    camera_normal = np.asarray(camera_normal.data)
    if camera_normal.shape[0] != 3:
        camera_normal = camera_normal[:3]
    proj = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    camera2d_from_camera3d = geo.Transform(proj, _inv=proj.T)
    # Initial vectors of the un-initialized camera matrices are converted to IJK space
    # R_volume.T is "RAS to IJK matrix", camera_intrinsic.inv is 2D to 3D
    initial_normal = camera2d_from_camera3d.inv @ camera_intrinsic.inv @ camera_intrinsic.optical_center
    initial_normal = geo.vector(initial_normal.__array__() / initial_normal.norm()).__array__()
    initial_z = camera2d_from_camera3d.inv @ camera_intrinsic.inv @ geo.vector(0, 1)
    initial_z = geo.vector(initial_z.__array__() / initial_z.norm()).__array__()
    final_z = (volume.world_from_ijk  @ geo.vector(0, 0, 1)).normalized().__array__() # We choose +K in volume to be aligned with +y in img
    # All the vectors should be in world space
    # Align the initial camera normal with the given camera_normal
    R_align = Rotation_to_align(initial_normal, camera_normal)

    # Align the initial camera z with the final_z
    new_z = R_align @ initial_z
    R_align = Rotation_to_align(new_z, final_z) @ R_align

    R_camera = geoR(R_align)
    return R_camera.data[:-1,:-1]



def apply_volume_rotation_to_camera_proj(camera_projection: CameraProjection, volume:Volume, rotation: np.ndarray,
                                         center: np.ndarray = None):
    """Applies a rotation to the camera projection, such that the camera is rotated around the volume center.
    This is equivalent to rotating the volume itself.
    Args:
        camera_projection (CameraProjection): The camera projection to rotate.
        volume (Volume): The volume to rotate around.
        rotation (np.ndarray): The rotation vector to apply (dim=3).
    """
    intrinsic = camera_projection.intrinsic
    extrinsic = camera_projection.extrinsic
    if center is None:
        center = volume.center_in_world
    T = FrameTransform.from_translation(center)
    new_extrinsic = extrinsic @ T @ FrameTransform.from_rotation(Rotation.from_rotvec(rotation).as_matrix()) @ T.inv
    new_camera_projection = CameraProjection(intrinsic, new_extrinsic)
    return new_camera_projection


def define_camera_matrix(volume: Volume, image_size: tuple, pixel_size: tuple, source_to_detector_distance: float,
                         source_to_isocenter_distance: float, center: np.ndarray, flip_up_down: bool=False,
                         camera_along_X: bool=False, source_posterior: bool=False):
    """Define camera position and orientation.
    camera_along_X: if False, camera will be oriented along the J axis of the volume in IJK space, else along the I axis.
    """
    camera_normal = (0., -1., 0.)
    camera_ray = geo.vector(*camera_normal)
    camera_ray_world = volume.world_from_ijk @ camera_ray
    camera_ray_world = geo.vector(camera_ray_world.__array__() / camera_ray_world.norm())
    cint = CameraIntrinsicTransform.from_sizes(image_size, pixel_size, source_to_detector_distance)
    T = geo.vector(volume.world_from_ijk @ geo.point(center)) + (-1. * camera_ray_world)* source_to_isocenter_distance
    R = R_from_IJK_to_C3D(volume, camera_ray_world, cint)
    cext = geo.FrameTransform.from_rt(R, geoR(R.T) @ T)
    if flip_up_down:
        Rupdown = Rotation.from_rotvec((0, np.pi, 0)).as_matrix().T
        cext = cext @ geoT(volume.world_from_ijk @ geo.point(center)) \
               @ geoR(Rupdown) @ geoT(volume.world_from_ijk @ geo.point(center)).inv
    cproj = CameraProjection(cint, cext)
    if source_posterior:
        cproj = apply_volume_rotation_to_camera_proj(cproj, volume, np.array([0, 0, np.pi]), 
                                                     center = volume.world_from_ijk @ geo.point(center))
    if camera_along_X:
        cproj = apply_volume_rotation_to_camera_proj(cproj, volume, np.array([0, 0, -np.pi * 90 / 180]),
                                                     center = volume.world_from_ijk @ geo.point(center))
    #
    #
    #
    #               ▲
    #               │   Volume point to be at the center
    #               │  +◄-----
    #               │   \   T \-----▲ N = normal . d_camera_volume
    #               │    \          │
    #               │     \\     \  │  /
    #               │       \\\   \ │ /
    #               │    Delta \\\ \│/
    #               │            \\\+
    #               │               Camera center
    #               │
    #               │Camera center needs to move by T = Delta - N
    #               │                   -1                      -1                -1
    #               │Ccenter = Extrinsic  . (0,0,0) -> Extrinsic  => T . Extrinsic
    #               │                             -1
    #               │-> Extrinsic => Extrinsic . T
    #               │
    #               │
    #               O─────────────────────────────────►
    #              //
    #             //
    #            /
    #          //
    #          /
    #         //
    #        //
    #       //
    #      //
    #     //
    #   //
    #  //
    # //
    # ▼
    #
    #
    #
    #
    # Correct the translation for the center of the camera to be aligned with the given center
    center_w = volume.world_from_ijk @ geo.point(center)
    cext = cproj.extrinsic
    delta = center_w - cproj.center_in_world
    proj = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    camera2d_from_camera3d = geo.Transform(proj, _inv=proj.T)
    normal = cext.inv @ camera2d_from_camera3d.inv @ cproj.intrinsic.inv @ cproj.intrinsic.optical_center
    normal = normal.normalized()
    normal = normal * source_to_isocenter_distance
    T = delta - normal
    cext = cext @ geoT(T).inv
    cproj = CameraProjection(cproj.intrinsic, cext)
    return cproj


def compute_mask_indices(mask, mask_world_from_ijk=None, vol_ijk_from_world=None):
    mask_indices = torch.argwhere(torch.as_tensor(mask) > 0.1)
    if mask_world_from_ijk is not None and vol_ijk_from_world is not None:
        vol_ijk_from_mask_ijk = torch.as_tensor(vol_ijk_from_world, dtype=torch.float64) @ torch.as_tensor(mask_world_from_ijk, dtype=torch.float64)
        mask_indices = torch.einsum('ab, nb->na', vol_ijk_from_mask_ijk, torch_to_point(mask_indices, dim=-1).to(vol_ijk_from_mask_ijk))[:,:-1]
    return mask_indices


def compute_mask_roi_from_indices(mask_indices):
    if not isinstance(mask_indices, np.ndarray):
        mask_indices = np.asarray(mask_indices)
    mask_roi = np.round(mask_indices.min(axis=0)).astype(int).tolist() + \
               np.round(mask_indices.max(axis=0) + 1).astype(int).tolist()
    return mask_roi


def roi3D_from_center(roi_size, center, volume):
    center = np.array(center)
    center_vox = np.array(volume.ijk_from_world @ geo.point(center))
    center_vox = np.round(center_vox).astype(int)
    roi_size = np.array(roi_size)
    roi_size_vox = roi_size / volume.spacing.__array__()
    img_roi = np.concatenate((center_vox - roi_size_vox / 2, center_vox + roi_size_vox / 2))
    img_roi = np.round(img_roi).astype(int)
    img_roi = np.concatenate((np.clip(img_roi[:3], np.zeros((3,)), volume.shape),
                                np.clip(img_roi[3:], np.zeros((3,)), volume.shape))).astype(int)
    center_vox = (img_roi[3:] + img_roi[:3]) // 2
    return center_vox,img_roi


def roi3D_from_segmentation(segmentation: nib.Nifti1Image, volume: Volume, img_roi_margin_vox: tuple, max_disp_vox: tuple, IJK_index: int):
    img_roi_margin_vox = np.asarray(img_roi_margin_vox)
    mask = torch.as_tensor(segmentation.get_fdata()).numpy()
    mask_indices = compute_mask_indices(mask, segmentation.affine, volume.ijk_from_world.data)  # in IJK
    mask_roi = compute_mask_roi_from_indices(mask_indices)
    print(f"Mask ROI: {mask_roi}")
    img_roi = np.concatenate((mask_roi[:3] - img_roi_margin_vox, mask_roi[3:] + img_roi_margin_vox))
    img_roi = np.round(img_roi).astype(int)
    img_roi = np.concatenate((np.clip(img_roi[:3], np.zeros((3,)), volume.shape),
                              np.clip(img_roi[3:], np.zeros((3,)), volume.shape))).astype(int)
    print(f"img_roi: {img_roi}")
    center = (img_roi[3:] + img_roi[:3]) // 2
    return center, img_roi


def disp_roi_from_img_roi(volume_shape, IJK_index, max_disp_vox, img_roi, allow_disp_outside_roi=False):
    img_roi = np.asarray(img_roi)
    max_disp_vox = np.asarray(max_disp_vox)
    roi_size = img_roi[3:] - img_roi[:3]
    if not allow_disp_outside_roi and (max_disp_vox > roi_size // 2).any():
        max_disp_vox = np.minimum(max_disp_vox, roi_size // 4)
        print("Warning: max_disp_vox is too large, it has been reduced to 1/4 the size of the roi," 
                "to prevent a null size disp_roi.")
    disp_roi = np.concatenate((img_roi[:3] + max_disp_vox, img_roi[3:] - max_disp_vox))
    disp_roi[IJK_index] = img_roi[IJK_index]
    disp_roi[IJK_index + 3] = img_roi[IJK_index + 3]
    disp_roi = np.round(disp_roi).astype(int)
    disp_roi = np.concatenate((np.clip(disp_roi[:3], np.zeros((3,)), volume_shape),
                              np.clip(disp_roi[3:], np.zeros((3,)), volume_shape))).astype(int)   
    print(f"disp_roi: {disp_roi}")                   
    return disp_roi
    

def torch_to_point(x, dim=-1):
    if dim==-1:
        return torch.concat((x, torch.ones_like(x[..., -1:])), dim=-1)
    elif dim==0:
        return torch.concat((x, torch.ones_like(x[-1:, ...])), dim=0)
    else:
        return torch.concat((x, torch.ones_like(x[(slice(None),)*dim +
                                                  (slice(x.shape[dim]-1, x.shape[dim], 1),) +
                                                  (...,)])), dim=dim)


def roi2D_from_roi3D(roi3D: list, camera_projection: CameraProjection, world_from_ijk, ijk_from_world, check_plot=False):
    wijk = torch.as_tensor(np.array(world_from_ijk)).to(torch.float64)
    ijkw = torch.as_tensor(np.array(ijk_from_world)).to(torch.float64)
    world_from_index = frame_tf_to_torch(camera_projection.world_from_index).to(torch.float64)
    world_from_camera3d = frame_tf_to_torch(camera_projection.world_from_camera3d).to(torch.float64)
    ray_origin = geo_to_torch(camera_projection.intrinsic.optical_center).to(torch.float64)
    camera_position = (ijkw @ torch.as_tensor(camera_projection.center_in_world.data).to(ijkw))[:-1]
    
    # Must be used with float64 else error of up to 1 ijk unit
    la, lab = get_torch_projection_ray(world_from_index, world_from_camera3d, ray_origin)
    # Here la, lab is converted from world to ijk
    lab = torch.einsum('ab, ...b->...a', ijkw, lab)
    la = torch.einsum('ab, ...b->...a', ijkw, la)
    boxes = torch_to_point(torch.as_tensor(roi3D).to(wijk).reshape(2,3), dim=-1)
    # 6 planes of the volume cube in IJK space are formed by the combinations of 3 base vectors and 2 points
    # Order of the planes : JKmin, IKmin, IJmin, JKmax, IKmax, IJmax
    plane_points_idx = torch.tensor([[(0,l[0],l[1]) for l in prod([0,1],[0,1])],
                                     [(l[0],0,l[1]) for l in prod([0,1],[0,1])],
                                     [(l[0],l[1],0) for l in prod([0,1],[0,1])],
                                     [(1,l[0],l[1]) for l in prod([0,1],[0,1])],
                                     [(l[0],1,l[1]) for l in prod([0,1],[0,1])],
                                     [(l[0],l[1],1) for l in prod([0,1],[0,1])],
                                    ])
    plane_points = make3DGrid((2,2,2), roi3D[:3], roi3D[3:], sparse=False).to(wijk).permute(1,2,3,0)
    p1p2 = torch.tensor([[(0,1,0), (1,0,0), (1,0,0)], [(0,0,1), (0,0,1), (0,1,0)]], dtype=la.dtype, device=la.device)
    planes_intersections = intersect_ray_plane_torch(la, lab, boxes, p1p2[0], p1p2[1]).squeeze()[..., :-1]
    planes_intersections_idx = torch.where(check_point_in_bb(boxes[...,:3], planes_intersections))[0]
    if len(planes_intersections_idx) != 2:
        raise ValueError(f"The camera principal ray must intersect the disp_roi 3D in 2 points, not {len(planes_intersections_idx)}")
    distances = torch.linalg.norm(planes_intersections[planes_intersections_idx] - camera_position, dim=-1)
    closest_plane_idx = planes_intersections_idx[distances.argmin()]
    farthest_plane_idx = planes_intersections_idx[distances.argmax()]
    # To get the biggest 2D roi while keeping all points of the 2D roi inside the 3D roi, 
    # select the corners of the farthest plane, make rays to the camera origin, intersect the rays with the closest plane and project in 2D
    farthest_points_idx = plane_points_idx[farthest_plane_idx]
    farthest_points = plane_points[farthest_points_idx[:,0],
                                   farthest_points_idx[:,1],
                                   farthest_points_idx[:,2]]
    rays = farthest_points - camera_position
    closest_points = intersect_ray_plane_torch(la, torch_to_vector(rays, dim=-1), boxes, p1p2[0], p1p2[1])[:,closest_plane_idx]
    
    if check_plot:
        fig, ax = make_3D_plot()
        ax.scatter(*farthest_points.T)
        ax.scatter(*camera_position, color='red')
        ax.plot(*torch.cat((farthest_points, camera_position.unsqueeze(0)), dim=0).T, color='blue')
        lines = [np.stack([camera_position.cpu().numpy(), corner]) for corner in farthest_points]
        for l in lines:
            ax.plot(*l.T, color='violet')
        ax.scatter(*closest_points.T)
        ax.plot(*plane_points[plane_points_idx[closest_plane_idx][:,0],
                    plane_points_idx[closest_plane_idx][:,1],
                    plane_points_idx[closest_plane_idx][:,2],].T)
        
    closest_points_2D = torch.einsum('ab,nb->na',
                                 torch.as_tensor(camera_projection.index_from_world.data).to(wijk) @ wijk,
                                 closest_points)
    closest_points_2D = closest_points_2D[:,:-1] / closest_points_2D[:,-1:]
    roi2D = closest_points_2D.amin(0).ceil().int().tolist() + closest_points_2D.amax(0).floor().int().tolist()
    roi2D = np.clip(roi2D, 0, camera_projection.intrinsic.sensor_size[0] - 1)
    return roi2D


def ConvertHUVolumeToDdrr(ddrr_volume, hu_volume):
    ddrr_volume.materials = ddrr_volume._format_materials(ddrr_volume._segment_materials(hu_volume))
    ddrr_volume.data = np.array(ddrr_volume._convert_hounsfield_to_density(hu_volume)).astype(np.float32)
    return ddrr_volume


def frame_tf_to_torch(mat, dtype=torch.float64, device=None):
    return torch.as_tensor(mat.data, dtype=dtype, device=device)


def world_from_ijk_in_roi(world_from_ijk: torch.Tensor, roi: list):
    """
    Returns the world_from_ijk matrix corresponding to the volume cropped to the given roi.
    """
    translation_offset = torch.as_tensor(np.array(roi[:3])).to(world_from_ijk)
    translation_offset = torch.einsum('ab, b...->a...', world_from_ijk,
                                      torch_to_point(translation_offset, dim=0))[:3]
    world_from_ijk_roi = world_from_ijk.clone()
    world_from_ijk_roi[:3, 3] = translation_offset
    return world_from_ijk_roi


class Crop:
    def __init__(self, indices=None, roi=None):
        if indices is None:
            if roi is None:
                raise ValueError("Either indices or roi must be specified.")
            else:
                indices = [r for i in range(0, len(roi)//2) for r in (roi[i], roi[i+len(roi)//2])]
        self.slice = [slice(indices[i], indices[i+1]) for i in range(0,len(indices)-1,2)]

    def __call__(self, img, pad=False):
        if pad:
            try:
                new_img = torch.zeros_like(img)
            except TypeError:
                new_img = np.zeros_like(img) #if TypeError again, call will just fail with error as intended
            new_img = new_img + img.min()
            new_img[(...,) + tuple(self.slice)] = img[(...,) + tuple(self.slice)]
            return new_img
        else:
            try:
                return img[(...,) + tuple(self.slice)]
            except TypeError:
                return img.slicer[(...,) + tuple(self.slice)]


def crop_ddrr_volume(volume: Volume, hu_volume: np.ndarray, roi: Union[tuple, np.ndarray, list, torch.Tensor]):
    # Cropping the volume requires to adjust the coordinate matrices through anatomical_from_ijk
    anatomical_from_ijk = frame_tf_to_torch(volume.anatomical_from_ijk)
    anatomical_from_ijk = world_from_ijk_in_roi(anatomical_from_ijk, roi)
    volume.anatomical_from_ijk = FrameTransform(anatomical_from_ijk.cpu().numpy())
    indices = [roi[0], roi[3], roi[1], roi[4], roi[2], roi[5]]
    hu_volume = Crop(indices)(hu_volume, pad=False)
    ConvertHUVolumeToDdrr(volume, hu_volume)
    return volume, hu_volume


def torch_pop(x, indices):
    mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    mask[indices] = False
    return x[mask]


def geo_to_torch(x):
    return torch.as_tensor(x.data)


def torch_to_point(x, dim=-1):
    if dim==-1:
        return torch.concat((x, torch.ones_like(x[..., -1:])), dim=-1)
    elif dim==0:
        return torch.concat((x, torch.ones_like(x[-1:, ...])), dim=0)
    else:
        return torch.concat((x, torch.ones_like(x[(slice(None),)*dim +
                                                  (slice(x.shape[dim]-1, x.shape[dim], 1),) +
                                                  (...,)])), dim=dim)


def torch_to_vector(x, dim=-1):
    if dim==-1:
        return torch.concat((x, torch.zeros_like(x[..., -1:])), dim=-1)
    elif dim==0:
        return torch.concat((x, torch.zeros_like(x[-1:, ...])), dim=0)
    else:
        raise ValueError
    

def make3DGrid(shape, mins=None, maxs=None, dtype=torch.float32, device=None, sparse=False, indexing='ij'):
    m, n, p = shape
    mi = mins if mins is not None else [0,0,0]
    ma = maxs if maxs is not None else [m-1,n-1,p-1]
    xd = torch.linspace(mi[0], ma[0], m, device=device, dtype=dtype)
    yd = torch.linspace(mi[1], ma[1], n, device=device, dtype=dtype)
    zd = torch.linspace(mi[2], ma[2], p, device=device, dtype=dtype)
    if sparse:
        return xd.reshape(m, 1, 1), yd.reshape(1, n, 1), zd.reshape(1, 1, p)
    return torch.stack(torch.meshgrid(xd, yd, zd, indexing=indexing))


def intersect_ray_plane_torch(la, lab, p0, p01, p02):
    p01xp02 = torch_to_vector(p01.cross(p02, dim=-1))
    lap0 = la.unsqueeze(1).expand((lab.shape[0],)+p0.shape) - p0.unsqueeze(0).expand((lab.shape[0],)+p0.shape)
    det = -torch.einsum('...b, cb->...c', lab, p01xp02)
    if torch.allclose(det, torch.zeros_like(det)):
        raise ValueError("Det of intersection should not be 0")
    t = (torch.einsum('...ac, ...c->...ac', torch.einsum('...ab, cb->...ac', lap0, p01xp02), 1. / det)).flatten(start_dim=1)
    return la.unsqueeze(1) + torch.einsum('...a, ...b->...ab', t, lab)


def get_torch_projection_ray(world_from_index, world_from_camera3d, ray_origin):
    ray_origin = ray_origin.unsqueeze(0) if len(ray_origin.shape) < 2 else ray_origin
    lab = torch.einsum('ab, ...b->...a', world_from_index, ray_origin)
    # la is the origin point of all rays, ie get_center_in_world
    la = torch.einsum('ab, ...b->...a', world_from_camera3d,
                      torch_to_point(torch.zeros((3,), dtype=lab.dtype, device=lab.device)))
    la = la[..., :-1] / la[..., -1].unsqueeze(-1)
    la = torch_to_point(la).unsqueeze(0)
    return la, lab


def filter_degenerate_elements(x, filter_value):
    if len(x.shape) >1:
        raise ValueError
    unique, counts = torch.unique_consecutive(x, return_counts=True)
    degenerate_elements = unique[torch.where(counts > filter_value)]
    degenerate_indices = [torch.where(x == d)[0][filter_value:] for d in degenerate_elements]
    return degenerate_elements, degenerate_indices


def intersect_ray_box_torch(ijk_from_world, volume_shape,
                            world_from_index, world_from_camera3d, ray_origin,
                            min_box=None, max_box=None, filter_degenerate=True):
    # Must be used with float64 else error of up to 1 ijk unit
    la, lab = get_torch_projection_ray(world_from_index, world_from_camera3d, ray_origin)
    # Here la, lab is converted from world to ijk
    lab = torch.einsum('ab, ...b->...a', ijk_from_world, lab)
    la = torch.einsum('ab, ...b->...a', ijk_from_world, la)
    min_box = min_box if min_box is not None else torch_to_point(torch.zeros(len(volume_shape), dtype=la.dtype, device=la.device))
    max_box = max_box if max_box is not None else torch_to_point(torch.tensor([v-1. for v in volume_shape], dtype=la.dtype, device=la.device))
    boxes = torch.stack((min_box, max_box))
    # 6 planes of the volume cube in IJK space are formed by the combinations of 3 base vectors and 2 points
    p1p2 = torch.tensor([[(0,1,0), (1,0,0), (1,0,0)], [(0,0,1), (0,0,1), (0,1,0)]], dtype=la.dtype, device=la.device)
    planes_intersections = intersect_ray_plane_torch(la, lab, boxes, p1p2[0], p1p2[1])
    # Checking which intersections points lie in the volume
    indices_intersections_in_box = torch.where(torch.logical_and(torch.less_equal(planes_intersections, max_box+1e-2).sum(-1) == planes_intersections.shape[-1],
                                                                 torch.greater_equal(planes_intersections, min_box - 1e-2).sum(-1) == planes_intersections.shape[-1]))
    intersection_ray_index = indices_intersections_in_box[0] #IDs of the rays that produced the intersections points
    box_intersections = planes_intersections[indices_intersections_in_box]
    box_intersections = box_intersections[..., :-1]
    if filter_degenerate:
        # Checking that each ray has only 2 intersections
        degenerate_ray_index, degenerate_ray_index_index = filter_degenerate_elements(intersection_ray_index, 2)
        if len(degenerate_ray_index) != 0:
            for indexes in degenerate_ray_index:
                intersection_ray_index = torch_pop(intersection_ray_index, indexes)
                box_intersections = torch_pop(box_intersections, indexes)
    return {'intersections': box_intersections, 'intersection_ray_index':intersection_ray_index, 'rays':(la,lab)}


def plot_img(projection,
             fax=None,
             title="", cmap='gray', colorbar=True, display=False, *args, **kwargs):
    if fax is None:
        f,ax=plt.subplots()
    else:
        f,ax=fax
    im = ax.imshow(projection, cmap=cmap, *args, **kwargs)
    ax.set_title(title)
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im, cax=cax, orientation='vertical')
    if display:
        plt.show()
    return f,ax


def make_3D_plot(fig=None, ax=None):
    fig = fig if fig is not None else plt.figure()
    ax = ax if ax is not None else fig.add_subplot(111, projection='3d')
    return fig, ax


def plot_grid3D_cube(grid_min, grid_max, fig=None, ax=None, color = 'red'):
    fig, ax = make_3D_plot(fig, ax)
    g = lambda i, j: np.stack((box_3D[:, i], box_3D[:, j]), 1)
    box_3D = make3DGrid((2, 2, 2), grid_min, grid_max)
    box_3D = box_3D.flatten(1).numpy()
    pairs = [[0,1], [0,2], [0,4], [4,6], [4,5], [6,2],
             [6,7], [2,3], [3,7], [7,5], [5,1], [1,3]]
    for p in pairs:
        ax.plot(*g(*p), color=color)
    return fig, ax


def Visualize(volume, camera_projection, disp_roi, disp_roi_2D):
    camera_position = geo_to_torch(volume.ijk_from_world @ camera_projection.center_in_world)[:-1].to(torch.float64)
    # Intersection points between the projection lines of the 2D roi and the volume
    p2d_corners = torch_to_point(torch.as_tensor(np.stack(np.meshgrid(
                                np.linspace(disp_roi_2D[0], disp_roi_2D[2]-1, 2),
                                np.linspace(disp_roi_2D[1], disp_roi_2D[3]-1, 2),
                                indexing='ij')).reshape(2, -1).T, dtype=torch.float64), dim=-1)
    p3d_corners = intersect_ray_box_torch(geo_to_torch(volume.ijk_from_world).to(torch.float64), 
                                          volume.shape,
                                          geo_to_torch(camera_projection.world_from_index).to(torch.float64), 
                                          geo_to_torch(camera_projection.world_from_camera3d).to(torch.float64),
                                          p2d_corners,
                                        #   min_box=torch_to_point(torch.tensor(disp_roi[:3], dtype=torch.float64)),
                                        #   max_box=torch_to_point(torch.tensor(disp_roi[3:], dtype=torch.float64))
                                                               )['intersections'].numpy()
    lines = [np.stack([camera_position.cpu().numpy(), corner]) for corner in p3d_corners]
    
    fig, ax = plot_grid3D_cube(disp_roi[:3], disp_roi[3:], color='orange')
    # fig, ax = plot_ray_box(disp_roi, fig, ax, color1='green', color2='green')  
    ax.plot(*camera_position, '.', color='red')
    for l in lines:
        ax.plot(*l.T, color='violet')
    ax.set_title(f"Disp roi 3D (orange), camera position(red), projection of the 2D roi in the volume (violet),\n"
                 f"The projection of the 2D roi in the volume should be fully contained in the disp roi 3D (orange)",
                 fontsize=10)
    return fig, ax


def check_disp_roi_invisible(volume, disp_roi, camera_projection, source_to_detector_distance, gamma, img_crop, p_volume_crop_3D):
    volume.data[disp_roi[0]:disp_roi[3], 
        disp_roi[1]:disp_roi[4], 
        disp_roi[2]:disp_roi[5]] = volume.data.max()
    proj_max = project(camera_projection, volume, source_to_detector_distance, gamma=gamma)
    plot_img(proj_max, title="projection volume cropped max")
    proj_max_crop = proj_max[img_crop[0]:img_crop[1], img_crop[2]:img_crop[3]]
    proj_max_pad = np.zeros_like(p_volume_crop_3D)
    proj_max_pad[img_crop[0]:img_crop[1], img_crop[2]:img_crop[3]] = proj_max_crop
    plot_img(proj_max_crop, title="projection volume cropped max crop")
    plot_img(proj_max_pad, title="projection volume cropped max crop pad")