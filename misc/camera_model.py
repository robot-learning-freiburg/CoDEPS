from __future__ import annotations  # Type hint in from_tensor()

from collections import OrderedDict

import torch
from torch import Tensor


class CameraModel():

    def __init__(self, img_width: int, img_height: int, fx: float, fy: float, cx: float, cy: float):
        assert img_width > 0, "img_width <= 0 is not allowed"
        assert img_height > 0, "img_height <= 0 is not allowed"
        assert fx >= 0, "fx < 0 is not allowed"
        assert fy >= 0, "fy < 0 is not allowed"
        assert cx >= 0, "cx < 0 is not allowed"
        assert cy >= 0, "cy < 0 is not allowed"

        self.image_size = {"width": img_width, "height": img_height}
        self.intrinsics = OrderedDict([("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)])

    def to_tensor(self) -> Tensor:
        return Tensor(list(self.intrinsics.values()))

    @classmethod
    def from_tensor(cls, img_width: int, img_height: int, intrinsics: Tensor) -> CameraModel:
        intrinsics_npy = intrinsics.detach().cpu().numpy()
        return CameraModel(img_width, img_height, intrinsics_npy[0], intrinsics_npy[1],
                           intrinsics_npy[2], intrinsics_npy[3])

    def get_scaled_model(self, scale_u, scale_v):
        return CameraModel(self.image_size["width"] * scale_u, self.image_size["height"] * scale_v,
                           self.intrinsics["fx"] * scale_u, self.intrinsics["fy"] * scale_v,
                           self.intrinsics["cx"] * scale_u, self.intrinsics["cy"] * scale_v)

    def get_scaled_model_image_size(self, width, height):
        scale_u = width / self.image_size["width"]
        scale_v = height / self.image_size["height"]
        return CameraModel(width, height, self.intrinsics["fx"] * scale_u,
                           self.intrinsics["fy"] * scale_v, self.intrinsics["cx"] * scale_u,
                           self.intrinsics["cy"] * scale_v)

    def get_image_point(self, x3d, y3d, z3d):
        """Computes the 2d image coordinates of the incoming 3d world point(s)
        :param x3d, y3d, z3d: x,y,z coordinates of the incoming 3d world point(s)
        :return: u,v coordinates of the computed 2d image point(s)
        """
        u2d = (x3d / z3d) * self.intrinsics["fx"] + self.intrinsics["cx"]
        v2d = (y3d / z3d) * self.intrinsics["fy"] + self.intrinsics["cy"]
        return u2d, v2d

    def get_viewing_ray(self, u2d, v2d):
        """Computes the viewing ray(s) of the incoming image point(s)
        :param u2d, v2d: u,v coordinates of the incoming image point(s)
        :return: ray_x, ray_y, ray_z: x,y,z coordinates of the unit vector(s) representing the
            outgoing ray(s)
        """
        # Compute a vector that points in the direction of the viewing ray (assuming a depth of 1)
        ray_x = (u2d - self.intrinsics["cx"]) / self.intrinsics["fx"]
        ray_y = (v2d - self.intrinsics["cy"]) / self.intrinsics["fy"]
        ray_z = 1.0

        # Compute the norm of the ray vector #
        norm = torch.sqrt(ray_x**2 + ray_y**2 + ray_z**2)

        # Normalize the ray to obtain a unit vector
        ray_x /= norm
        ray_y /= norm
        ray_z /= norm

        return ray_x, ray_y, ray_z
