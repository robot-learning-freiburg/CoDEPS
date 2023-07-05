from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from misc.camera_model import CameraModel


class _PointcloudToImage(nn.Module):
    """Reprojects all pointclouds of the batch into the image and returns a new batch of the
    corresponding 2d image points
    """

    def __init__(self, img_width: int, img_height: int):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height

    def forward(self, batch_camera_models: List[CameraModel], batch_pcl):
        # Get the data from batch of depth images (1 dim for batch, 2 dims for matrix and 1 for
        # x,y,z values -> dims() = 4)
        # Attention,
        assert batch_pcl.dim() == 4, \
            f"The input pointcloud has {batch_pcl.dim} dimensions which is != 4"
        assert batch_pcl.size(1) == 3, \
            f"The input pointcloud has {batch_pcl.size(1)} channels which is != 3"

        batch_size = batch_pcl.size(0)
        x3d = batch_pcl[:, 0, :, :].view(batch_size, -1)
        y3d = batch_pcl[:, 1, :, :].view(batch_size, -1)
        z3d = batch_pcl[:, 2, :, :].clamp(min=1e-5).view(batch_size, -1)

        # Compute the pixel coordinates
        u2d = torch.empty_like(x3d)
        v2d = torch.empty_like(y3d)
        for i, camera_model in enumerate(batch_camera_models):
            u2d_, v2d_ = camera_model.get_image_point(x3d[i], y3d[i], z3d[i])
            u2d[i] = u2d_
            v2d[i] = v2d_
        # u2d, v2d = camera_model.get_image_point(x3d, y3d, z3d)

        # Normalize the coordinates to [-1,+1] as required for grid_sample
        u2d_norm = (u2d / (self.img_width - 1) - 0.5) * 2
        v2d_norm = (v2d / (self.img_height - 1) - 0.5) * 2

        # Put the u2d_norm and v2d_norm vectors together and reshape them
        pixel_coordinates = torch.stack([u2d_norm, v2d_norm], dim=2)  # dim: batch_size, H*W, 2
        pixel_coordinates = pixel_coordinates.view(batch_size, self.img_height, self.img_width, 2)

        return pixel_coordinates


class _ImageToPointcloud(nn.Module):
    """Projects all images of the batch into the 3d world (batch of pointclouds)
    """

    def __init__(self, img_width: int, img_height: int, device: torch.device):
        super().__init__()
        # Define a grid of pixel coordinates for the corresponding image size. Each entry defines
        # specific grid pixel coordinates for which the viewing ray is to be computed
        self.u2d_vals = torch.arange(start=0, end=img_width).expand(img_height,
                                                                    img_width).float().to(device)
        self.v2d_vals = torch.arange(start=0, end=img_height).expand(img_width,
                                                                     img_height).t().float().to(
            device)

    def forward(self, batch_camera_models: List[CameraModel], batch_depth):
        assert batch_depth.dim() == 4, \
            f"The input batch of depth maps has {batch_depth.dim()} dimensions which is != 4"
        assert batch_depth.size(1) == 1, \
            f"The input batch of depth maps has {batch_depth.size(1)} channels which is != 1"

        rays_x = torch.empty_like(batch_depth)
        rays_y = torch.empty_like(batch_depth)
        rays_z = torch.empty_like(batch_depth)
        for i, camera_model in enumerate(batch_camera_models):
            rays_x_, rays_y_, rays_z_ = camera_model.get_viewing_ray(self.u2d_vals, self.v2d_vals)
            rays_x[i, 0] = rays_x_
            rays_y[i, 0] = rays_y_
            rays_z[i, 0] = rays_z_

        x3d = batch_depth / abs(rays_z) * rays_x
        y3d = batch_depth / abs(rays_z) * rays_y
        z3d = batch_depth / abs(rays_z) * rays_z

        return torch.cat((x3d, y3d, z3d), dim=1)


class CoordinateWarper(nn.Module):

    def __init__(self, img_width: int, img_height: int, device: torch.device):
        super().__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.device = device
        self.image_to_pointcloud = _ImageToPointcloud(img_width, img_height, device)
        self.pointcloud_to_image = _PointcloudToImage(img_width, img_height)

    def forward(self, batch_camera_models: List[CameraModel], batch_depth_map, T,
                object_motion_map=None):
        """
        :param pointcloud: batch of point clouds
        :param T: Transformation matrix
        :return:
        Attention: R and t together denote a transformation rule which transforms the point cloud
        from the camera coordinate system of the source image to that one of the target image.
        So it's the inverted transformation from the source pose to the target pose
        """
        assert batch_depth_map.dim() == 4, \
            f"The input batch of depth maps has {batch_depth_map.dim()} dimensions which is != 4"
        assert batch_depth_map.size(1) == 1, \
            f"The input batch of depth maps has {batch_depth_map.size(1)} channels which is != 1"

        # Reproject all image pixel coordinates into the 3d world (pointcloud)
        image_as_pointcloud = self.image_to_pointcloud(batch_camera_models, batch_depth_map)

        # Transform the pointcloud to homogeneous coordinates
        ones = nn.Parameter(torch.ones(batch_depth_map.size(0),
                                       1,
                                       self.img_height,
                                       self.img_width,
                                       device=self.device),
                            requires_grad=False)
        image_as_pointcloud_homogeneous = torch.cat([image_as_pointcloud, ones], 1)

        # Transform the obtained pointcloud into the local coordinate system of the target camera
        # pose (homogeneous)
        transformed_pointcloud = torch.bmm(
            T, image_as_pointcloud_homogeneous.view(batch_depth_map.size(0), 4, -1))
        transformed_pointcloud = transformed_pointcloud.view(-1, 4, self.img_height,
                                                             self.img_width)
        if object_motion_map is not None:
            transformed_pointcloud[:, :-1, :, :] += object_motion_map

        # Transform back to Euclidean coordinates
        transformed_pointcloud = transformed_pointcloud[:, :3, :, :] / \
                                 transformed_pointcloud[:, 3, :, :].unsqueeze(1)

        # Compute pixel_coordinates, which includes associations from each pixel of the source image
        # to the target image
        pixel_coordinates = self.pointcloud_to_image(batch_camera_models, transformed_pointcloud)

        return pixel_coordinates


class ImageWarper(nn.Module):

    def __init__(self, img_width: int, img_height: int, device: torch.device):
        super().__init__()
        self.coordinate_warper = CoordinateWarper(img_width, img_height, device)

    def forward(self, batch_camera_models: List[CameraModel], batch_src_img, batch_depth_map, T,
                interp_mode="bilinear", object_motion_map=None):
        """
        :param pointcloud: batch of point clouds
        :param interp_mode: interpolation mode of grid sampling
        :param T: Transformation matrix
        :return:
        Attention: R and t together denote a transformation rule which transforms the point cloud
        from the camera coordinate system of the source image to that one of the target image. So
        it's the inverted transformation from the source pose to the target pose
        """
        assert batch_src_img.dim() == 4, \
            f"The input batch of source images has {batch_src_img.dim()} dimensions which is != 4"
        # assert batch_src_img.size(1) == 3, \
        #     f"The input batch of source images has {batch_src_img.size(1)} channels which is != 3"

        pixel_coordinates = self.coordinate_warper(batch_camera_models, batch_depth_map, T,
                                                   object_motion_map)

        # ToDo: Here we use as padding mode "border" to account for pixels that are out of boundary.
        #  We could actually detach them completely from the computation graph (not very clever
        #  either...) Using border is not very useful, as the outer regions being padded are quite
        #  big. Detaching them may help...
        # Set align_corners to True for better performance
        # https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/8
        warped_image = F.grid_sample(batch_src_img,
                                     pixel_coordinates,
                                     mode=interp_mode,
                                     padding_mode="border",
                                     align_corners=True)

        return warped_image
