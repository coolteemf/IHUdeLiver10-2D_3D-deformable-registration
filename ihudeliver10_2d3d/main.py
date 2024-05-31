import argparse
import json
import os
from copy import deepcopy

import numpy as np
import nibabel as nib
from skimage import io
import matplotlib.pyplot as plt
from utils import Visualize, check_disp_roi_invisible, disp_roi_from_img_roi, roi2D_from_roi3D, roi3D_from_center, roi3D_from_segmentation, \
                  load_volume_data, define_camera_matrix, project, crop_ddrr_volume, \
                  crop_roi, crop_from_roi_2D, plot_img
import killeengeo as geo

def main(volume_path: str, 
         segmentation_path: str, 
         cuda_path: str,
         seg_margin: tuple,
         roi_size: tuple,
         max_disp: float,
         center: tuple, 
         name: str, 
         source_to_detector_distance: float, 
         source_to_isocenter_distance: float,
         detector_size: tuple, 
         pixel_size: tuple, 
         projection_size: tuple,
         camera_along_X: bool, 
         gamma: float, 
         flip_up_down: bool,
         display: bool,
         save: bool,
         source_posterior: bool,
         allow_disp_outside_roi: bool):
    ### CUDA path needs to be in path
    os.environ['PATH'] = f'{cuda_path}:' + os.environ['PATH']
    ### Check that the segmentation is aligned to volume
    ### If the segmentation of the organ is incorrect, as can be the case with pigs,
    ### it is possible to use a custom ROI instead
    ### The data will be saved in the same directory as the volume
    # Margin between segmentation and img_roi, in mm
    # Physical detector dimensions
    # Project a smaller image to speed up the process and reduce memory usage
    downscale_factor = detector_size[0] / projection_size[0]
    psize_resize = (pixel_size[0] * downscale_factor,) * 2  # update the pixel size to account for downscale
    if center is None:
        print("WARNING: center is not supplied, organ segmentation will be used to compute the roi automatically.")

    assert os.path.exists(volume_path)
    # If no name is provided, use the name of the volume
    volume_name = os.path.basename(volume_path).split(".nii")[0]
    if name is None:
        name = volume_name
    save_path = "/".join(volume_path.split("/")[:-1])
    
    nii_volume = load_volume_data(volume_path)
    hu_volume = nii_volume.get_fdata()
    volume = Volume.from_nifti(volume_path)

    IJK_index = 0 if camera_along_X else 1
    # Compute the img_roi which will contain the voxels to be used for the deformation
    # and the disp_roi in which the displacements will be observed
    max_disp_vox = np.ones(3) * max_disp / volume.spacing.__array__()
    if center is None:
        assert seg_margin is not None
        seg_margin_vox = np.array(seg_margin) / volume.spacing.__array__()
        assert os.path.exists(segmentation_path)
        center_vox, img_roi = roi3D_from_segmentation(
            load_volume_data(segmentation_path),
            volume, seg_margin_vox, max_disp_vox, IJK_index)    
    else:
        center_vox, img_roi = roi3D_from_center(roi_size, center, volume)
        
    center = np.array(volume.world_from_ijk @ geo.point(center_vox))
    #TODO change this so that disp roi stops at the edge of the issue for the posterior plane, to avoid deforming the CT table
    disp_roi = disp_roi_from_img_roi(volume.shape, 
                                     IJK_index, 
                                     max_disp_vox, 
                                     img_roi,
                                     allow_disp_outside_roi)

    cproj = define_camera_matrix(volume=volume, image_size=projection_size, pixel_size=psize_resize,
                                 source_to_detector_distance=source_to_detector_distance,
                                 source_to_isocenter_distance=source_to_isocenter_distance,
                                 center=center_vox, flip_up_down=flip_up_down,
                                 camera_along_X=camera_along_X, source_posterior=source_posterior)

    p_original = project(cproj, volume, source_to_detector_distance, gamma=gamma)
    # Crop the volume to the visible part to speed up projection and reduce memory usage
    volume, hu_volume = crop_ddrr_volume(volume, hu_volume, img_roi)
    disp_roi_uncrop = deepcopy(disp_roi)
    disp_roi = crop_roi(disp_roi, img_roi)
    p_volume_crop_3D = project(cproj, volume, source_to_detector_distance, gamma=gamma)

    # The roi_2D is defined as the biggest roi such that no points outside the disp_roi are in it
    # The camera projection is cropped to the roi_2D
    roi_2D = roi2D_from_roi3D(disp_roi, cproj, volume.world_from_ijk, volume.ijk_from_world)
    img_crop = crop_from_roi_2D(roi_2D)
    p_crop = p_volume_crop_3D[img_crop[0]:img_crop[1], img_crop[2]:img_crop[3]]
    p_pad = np.zeros_like(p_volume_crop_3D)
    p_pad[img_crop[0]:img_crop[1], img_crop[2]:img_crop[3]] = p_crop
    # If files already exist, increment the index
    it = 0
    while os.path.exists(os.path.join(save_path, f"{name}_intrinsic_{it}.npy")):
        it += 1
    
    if display:
        plot_img(p_original, title=f"original projection")
        plot_img(p_volume_crop_3D, title="projection volume cropped")
        plot_img(p_crop, title="projection 2D cropped")
        plot_img(p_pad, title="projection 2D cropped padded")
        Visualize(volume, cproj, disp_roi, roi_2D)
        # check_disp_roi_invisible(volume, disp_roi, cproj, source_to_detector_distance, gamma, img_crop, p_volume_crop_3D)
        plt.show()

    if save:
        np.save(os.path.join(save_path, f"{name}_intrinsic_{it}.npy"), cproj.intrinsic.data)
        np.save(os.path.join(save_path, f"{name}_extrinsic_{it}.npy"), cproj.extrinsic.data)
        np.save(os.path.join(save_path, f"{name}_worldfromanat_{it}.npy"), volume.world_from_anatomical.data)
        io.imsave(os.path.join(save_path, f"{name}_poriginal_{it}.png"),
                  np.round(p_original * 255).astype(np.uint8))
        io.imsave(os.path.join(save_path, f"{name}_pimgcrop_{it}.png"),
                  np.round(p_pad * 255).astype(np.uint8))
        io.imsave(os.path.join(save_path, f"{name}_pvolumecrop_{it}.png"),
                  np.round(p_volume_crop_3D * 255).astype(np.uint8))
        with open(os.path.join(save_path, f"{name}_projection_parameters_{it}.json"), 'w') as f:
            json.dump({"source_to_detector_distance": source_to_detector_distance,
                       "source_to_isocenter_distance": source_to_isocenter_distance,
                       "detector_size": detector_size,
                       "pixel_size": pixel_size,
                       "projection_size": projection_size,
                       "camera_along_X": camera_along_X,
                       "gamma": gamma,
                       "flip_up_down": flip_up_down,
                       "volume_path": volume_path,
                       "segmentation_path": segmentation_path,
                       "name": name,
                       "volume_name": volume_name,
                       "img_roi": img_roi.tolist(),
                       "disp_roi": disp_roi_uncrop.tolist(),
                       "roi_2D": roi_2D.tolist(),
                       "pixel_size_resize": psize_resize,
                       "intrinsic": cproj.intrinsic.data.tolist(),
                       "extrinsic": cproj.extrinsic.data.tolist(),
                       "world_from_anat": volume.world_from_anatomical.data.tolist()}, 
                      f,
                      indent=0)

    print(f"img_roi: {img_roi}\nroi_2D: {roi_2D}\ndisp_roi: {disp_roi_uncrop}\n"
          f"pixel_size: {psize_resize}\n"
          f"iteration {it}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a C-arm projection geometry automatically '
                                                 'using the CT geometry and input parameters.')
    parser.add_argument('--volume_path', type=str, required=True,
                        help='Path to the volume.')
    parser.add_argument('--cuda_path', type=str, required=True,
                        help='Path to the cuda executable.')
    parser.add_argument('--max_disp', type=float, required=True, default=150.,
                    help='Maximum displacement, controls size of the image (in mm).')
    parser.add_argument('--segmentation_path', type=str,
                        help='Path to the segmentation.')
    parser.add_argument('--center', nargs='+', type=float, default=None,
                    help='Center of the ROI (in world coordinates)')
    parser.add_argument('--roi_size', nargs='+', type=float, default=(200., 100., 200.),
                        help='Size of the roi (in mm).')
    parser.add_argument('--seg_margin', nargs='+', type=float, default=(100., 50., 100.),
                        help='Distance between the bounds of the organ segmentation and the bounds of the roi.')
    parser.add_argument('--name', type=str, default=None,
                        help='Name of the output.')
    parser.add_argument('--source_to_detector_distance', type=float, default=1500,
                        help='The distance between the X-ray source and the detector.')
    parser.add_argument('--source_to_isocenter_distance', type=float, default=1050,
                        help='The distance from the source to the isocenter.')
    parser.add_argument('--detector_size', nargs='+', type=int, default=(2330, 2330),
                        help='Size of the detector.')
    parser.add_argument('--pixel_size', nargs='+', type=float, default=(0.148, 0.148),
                        help='Size of the pixel.')
    parser.add_argument('--projection_size', nargs='+', type=int, default=(512, 512),
                        help='Size of the projection.')
    parser.add_argument('--camera_along_X', action=argparse.BooleanOptionalAction, default=False,
                        help='Is the camera pointed along X (first voxel axis) ?')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Gamma value.')
    parser.add_argument('--flip_up_down', action=argparse.BooleanOptionalAction, default=False,
                        help='Flip up down?')
    parser.add_argument('--display', action=argparse.BooleanOptionalAction, default=True,
                        help='Display the output?')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True,
                        help='Save the output?')
    parser.add_argument('--source_posterior', action=argparse.BooleanOptionalAction, default=False,
                        help='Is the source on the posterior side?')
    parser.add_argument('--allow_disp_outside_roi', action=argparse.BooleanOptionalAction, 
                        default=False,
                        help='Allow displacements outside the roi (may lead to img_roi crop visible in the image)')
    args = parser.parse_args()

    main(args.volume_path, args.segmentation_path, args.cuda_path, args.seg_margin,
         args.roi_size, args.max_disp, args.center, args.name,
         args.source_to_detector_distance, args.source_to_isocenter_distance,
         args.detector_size, args.pixel_size, args.projection_size,
         args.camera_along_X, args.gamma, args.flip_up_down, args.display, args.save,
         args.allow_disp_outside_roi,
         args.source_posterior)
