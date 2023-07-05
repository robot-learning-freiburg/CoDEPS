# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from collections import OrderedDict
from typing import List

import torch
from torch import Tensor, nn


class PoseHead(nn.Module):

    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features: List[Tensor], invert_pose: bool = False) -> Tensor:
        last_features = input_features[-1]

        out = self.relu(self.convs["squeeze"](last_features))
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        # ToDo: Why
        pose_diff = self.transformation_from_parameters(axisangle[:, 0], translation[:, 0],
                                                        invert_pose)
        return pose_diff

    @staticmethod
    def transformation_from_parameters(axisangle: Tensor,
                                       translation: Tensor,
                                       invert: bool = False) -> Tensor:
        """Convert the network's (axisangle, translation) output into a 4x4 matrix
        # Source: https://github.com/nianticlabs/monodepth2/blob/master/layers.py
        """
        R = PoseHead.rot_from_axisangle(axisangle)
        t = translation.clone()

        if invert:
            R = R.transpose(1, 2)
            t *= -1

        T = PoseHead.get_translation_matrix(t)

        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)

        return M

    @staticmethod
    def rot_from_axisangle(axisangle: Tensor) -> Tensor:
        """Convert an axisangle rotation into a 4x4 transformation matrix
        # Source: https://github.com/Wallacoloo/printipi (adapted by Monodepth2)
        Input 'vec' has to be Bx1x3
        """
        angle = torch.norm(axisangle, 2, 2, True)
        axis = axisangle / (angle + 1e-7)

        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca

        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)

        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC

        rot = torch.zeros((axisangle.shape[0], 4, 4), device=axisangle.device)

        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1

        return rot

    @staticmethod
    def get_translation_matrix(translation_vector: Tensor) -> Tensor:
        """
        Source: https://github.com/nianticlabs/monodepth2/blob/master/layers.py
        Convert a translation vector into a 4x4 transformation matrixF
        """
        T = torch.zeros(translation_vector.shape[0], 4, 4, device=translation_vector.device)

        t = translation_vector.contiguous().view(-1, 3, 1)

        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t

        return T
