from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.contrib import distance_transform
from numpy.typing import ArrayLike
from torch import Tensor
from yacs.config import CfgNode as CN

from algos import InstanceSegAlgo
from datasets.dataset import Dataset
from datasets.preprocessing import transfer_histogram_style
from misc.camera_model import CameraModel
from misc.image_warper import _ImageToPointcloud


# ToDo: The class has no attributes atm. Maybe just make a bunch of functions?
# ToDo: Case handling as usual, Nones and parameters that are not set etc...
# ToDo: We assume a batch size of 1 for each mixup operation. But this is actually fine
class Mixup:
    def __init__(self):
        pass
        # self.image_to_pointcloud = _ImageToPointcloud(self.img_width_tgt, self.img_height_tgt,
        #                                               device)

    @staticmethod
    def _src_pcl_to_tgt(batch_camera_models: List[CameraModel], img_shape_src: Tuple[int, int],
                        img_shape_tgt: Tuple[int, int], batch_pcl: torch.Tensor):
        # Get the data from batch of depth images (1 dim for batch, 2 dims for matrix and 1 for
        # x,y,z values -> dims() = 4)
        # Attention,
        assert batch_pcl.dim() == 4, \
            f"The input pointcloud has {batch_pcl.dim} dimensions which is != 4"
        assert batch_pcl.size(1) == 3, \
            f"The input pointcloud has {batch_pcl.size(1)} channels which is != 3"

        # Reassign variables
        img_width_src = img_shape_src[-1]
        img_height_src = img_shape_src[-2]
        img_width_tgt = img_shape_tgt[-1]
        img_height_tgt = img_shape_tgt[-2]

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

        # Normalize the coordinates to [-1,+1] as required for grid_sample
        u2d_norm = (u2d / (img_width_src - 1) - 0.5) * 2
        v2d_norm = (v2d / (img_height_src - 1) - 0.5) * 2

        # Put the u2d_norm and v2d_norm vectors together and reshape them
        pixel_coordinates = torch.stack([u2d_norm, v2d_norm], dim=2)  # dim: batch_size, H*W, 2
        pixel_coordinates = pixel_coordinates.view(batch_size, img_height_tgt, img_width_tgt, 2)

        return pixel_coordinates

    @staticmethod
    def _get_rnd_cls(lbl_sem: torch.Tensor):
        all_classes = torch.unique(lbl_sem)
        num_classes = all_classes.shape[0]
        return (all_classes[torch.Tensor(np.random.choice(num_classes,
                                                          int((num_classes + num_classes % 2) / 2),
                                                          replace=False)).long()])

    @staticmethod
    def _gen_class_msk(lbl_sem: torch.Tensor, classes: torch.Tensor):
        lbl_sem_cast, classes_cast = torch.broadcast_tensors(lbl_sem,
                                                             classes.unsqueeze(1).unsqueeze(2))
        return torch.eq(lbl_sem_cast, classes_cast).sum(1)

    @staticmethod
    def _get_cls_msk(lbl_sem: torch.Tensor):
        rnd_cls = Mixup._get_rnd_cls(lbl_sem)
        cls_msk = Mixup._gen_class_msk(lbl_sem, rnd_cls)
        return cls_msk

    @staticmethod
    def _get_cut_msk(nof_vert_split: int, nof_hor_split: int, img_tgt: Tensor, img_src: Tensor,
                     semantic_pred_tgt_ema: Tensor, tgt_is_replay: Tensor, nof_segments: int = 2):
        if img_tgt.shape == img_src.shape:
            msk_lst = []
            # Iterate over batch
            for b in range(img_tgt.shape[0]):
                height, width = img_tgt.shape[-2:]
                nof_splits = nof_vert_split * nof_hor_split

                crop_height = height // nof_vert_split
                crop_width = width // nof_hor_split

                msk = torch.zeros((1, img_tgt.shape[-2], img_tgt.shape[-1]), dtype=torch.uint8,
                                  device=img_tgt.device)

                if tgt_is_replay[b] == 0 or True:
                    # For online, take a random patch:
                    split_ids = np.random.choice(nof_splits, nof_segments, replace=False)
                else:
                    # For replay, use a variant of rare class sampling

                    # ToDo: Hardcoded for now
                    inv_frequency = np.array(
                        [2.69302949, 16.30920132, 4.23122285, 113.12797861, 80.92451295,
                         180.22584596, 6.23900793, 85.71399802, 24.77534557, 81.45685098,
                         734.90701854, 14.19108126, 371.23855234, 193.67350238])
                    inv_frequency /= inv_frequency.max()
                    # T = 0.5

                    scores = [0] * nof_splits
                    for split_id in range(nof_splits):
                        mul_x = split_id % nof_hor_split
                        mul_y = split_id // nof_hor_split
                        crop_x_strt, crop_x_end = mul_x * crop_width, (mul_x + 1) * crop_width - 1
                        crop_y_strt, crop_y_end = mul_y * crop_height, (mul_y + 1) * crop_height - 1
                        labels, counts = semantic_pred_tgt_ema[b][crop_y_strt:crop_y_end,
                                         crop_x_strt:crop_x_end].unique(return_counts=True)
                        labels = labels.cpu().numpy()
                        counts = inv_frequency[labels] * counts.cpu().numpy()
                        # scores[split_id] = np.exp(counts[labels >= 9].sum() / counts.sum() / T) / (np.exp(counts[labels >= 9].sum() / counts.sum() / T) + np.exp(counts[labels < 9].sum() / counts.sum() / T))
                        # counts = counts.cpu().numpy()
                        scores[split_id] = max(counts[labels >= 9].sum() / counts.sum(), .01)
                    # scores = [s / sum(scores) for s in scores]

                    split_ids = np.random.choice(nof_splits, nof_segments, replace=False, p=scores)

                for split_id in split_ids:
                    mul_x = split_id % nof_hor_split
                    mul_y = split_id // nof_hor_split

                    crop_x_strt, crop_x_end = mul_x * crop_width, (mul_x + 1) * crop_width - 1
                    crop_y_strt, crop_y_end = mul_y * crop_height, (mul_y + 1) * crop_height - 1

                    # Create msk
                    msk[:, crop_y_strt:crop_y_end, crop_x_strt:crop_x_end] = 1

                msk_lst.append(msk.bool())

            out_mask = torch.stack(msk_lst, dim=0)
            return out_mask, out_mask

        assert False, "Not implemented"
        msk_src_lst = []
        msk_tgt_lst = []
        # Iterate over batch
        for _ in range(img_tgt.shape[0]):
            height_tgt, width_tgt = img_tgt.shape[-2:]
            height_src, width_src = img_src.shape[-2:]
            nof_splits = nof_vert_split * nof_hor_split

            crop_height_tgt = height_tgt // nof_vert_split
            crop_width_tgt = width_tgt // nof_hor_split
            crop_height_src = height_src // nof_vert_split
            crop_width_src = width_src // nof_hor_split

            msk_tgt = torch.zeros((1, img_tgt.shape[-2], img_tgt.shape[-1]), dtype=torch.uint8,
                                  device=img_tgt.device)
            msk_src = torch.zeros((1, img_src.shape[-2], img_src.shape[-1]), dtype=torch.uint8,
                                  device=img_src.device)

            split_ids = np.random.choice(nof_splits, nof_segments, replace=False)
            for split_id in split_ids:
                mul_x = split_id % nof_hor_split
                mul_y = split_id // nof_hor_split

                crop_x_strt_tgt, crop_x_end_tgt = mul_x * crop_width_tgt, (
                        mul_x + 1) * crop_width_tgt - 1
                crop_y_strt_tgt, crop_y_end_tgt = mul_y * crop_height_tgt, (
                        mul_y + 1) * crop_height_tgt - 1

                # Create tgt msk
                msk_tgt[:, crop_y_strt_tgt:crop_y_end_tgt, crop_x_strt_tgt:crop_x_end_tgt] = 1

                # Create src msk
                x_src_min, x_src_max = mul_x * crop_width_src, (mul_x + 1) * crop_width_src - 1
                crop_x_strt_src = int(np.random.uniform(low=x_src_min, high=x_src_max))
                crop_x_end_src = crop_x_strt_src + crop_width_tgt - 1
                offset_x = width_src - crop_x_end_src - 1
                if crop_x_end_src > width_src - 1:
                    crop_x_end_src += offset_x
                    crop_x_strt_src += offset_x

                y_src_min, y_src_max = mul_y * crop_height_src, (mul_y + 1) * crop_height_src - 1
                crop_y_strt_src = int(np.random.uniform(low=y_src_min, high=y_src_max))
                crop_y_end_src = crop_y_strt_src + crop_height_tgt - 1
                offset_y = height_src - crop_y_end_src - 1
                if crop_y_end_src > height_src - 1:
                    crop_y_end_src += offset_y
                    crop_y_strt_src += offset_y

                msk_src[:, crop_y_strt_src:crop_y_end_src, crop_x_strt_src:crop_x_end_src] = 1

            # Cast to boolean
            msk_src_lst.append(msk_src.bool())
            msk_tgt_lst.append(msk_tgt.bool())

        out_msk_src = torch.stack(msk_src_lst, dim=0)
        out_msk_tgt = torch.stack(msk_tgt_lst, dim=0)

        return out_msk_src, out_msk_tgt

    @staticmethod
    def warp_c2c(cam_model_src: List[CameraModel], cam_model_tgt: List[CameraModel],
                 in_src: torch.Tensor, in_tgt: torch.Tensor, depth_val: Optional[float] = 1,
                 interp_mode="bilinear", padding_mode="border"):
        # ToDo: Here we put back the 'channel' dimension for consistency
        if in_src.dim() == 3:
            in_src = in_src.unsqueeze(1)

        # ToDo: Check whether depth matters here
        image_to_pointcloud = _ImageToPointcloud(in_tgt.shape[-1], in_tgt.shape[-2],
                                                 in_src.device)

        # Create dummy depth map
        depth_map = torch.ones_like(in_tgt)[:, 0, :, :].unsqueeze(1).double() * depth_val
        pcl_tgt = image_to_pointcloud(cam_model_tgt, depth_map)
        pixel_coords = Mixup._src_pcl_to_tgt(cam_model_src, in_src.shape, in_tgt.shape, pcl_tgt)
        warped_img = F.grid_sample(in_src.double(), pixel_coords.double(), mode=interp_mode,
                                   padding_mode=padding_mode, align_corners=True)

        return warped_img

    def embed_wsrc2tgt(self, cam_model_src: List[CameraModel], cam_model_tgt: List[CameraModel],
                       img_src: torch.Tensor, img_tgt: torch.Tensor):
        warped_img_src = self.warp_c2c(cam_model_src, cam_model_tgt, img_src, img_tgt,
                                       padding_mode="zeros")
        msk = warped_img_src == 0

        joint_img = warped_img_src
        joint_img[msk] = img_tgt[msk]

        return joint_img

    @staticmethod
    def get_off_cen(lbl_inst_mixup: torch.Tensor):
        lbl_offset_mixup_lst = []
        lbl_center_mixup_lst = []
        # ToDo: Copying back to cpu is super bad
        for lbl_inst_mixup_i in lbl_inst_mixup:
            offset_gt_mixup_np_i, center_gt_mixup_np_i = Dataset.get_offset_center(
                np.asarray(lbl_inst_mixup_i.squeeze().cpu()))
            lbl_offset_mixup_lst.append(
                torch.from_numpy(offset_gt_mixup_np_i).to(lbl_inst_mixup.device))
            lbl_center_mixup_lst.append(
                torch.from_numpy(center_gt_mixup_np_i).to(lbl_inst_mixup.device))
        lbl_offset_mixup = torch.stack(lbl_offset_mixup_lst, dim=0)
        lbl_center_mixup = torch.stack(lbl_center_mixup_lst, dim=0)

        return lbl_center_mixup, lbl_offset_mixup

    # This is a naive implementation with image resizing
    @staticmethod
    def class_mixup(img_src: torch.Tensor, img_tgt: torch.Tensor,
                    lbl_instance_src: torch.Tensor, lbl_sem_src: torch.Tensor,
                    semantic_pred_tgt_ema, instance_pred_tgt_ema, geom_augment, conf_thresh,
                    get_off_cen=True):

        lbl_instance_tgt = instance_pred_tgt_ema.clone()
        lbl_sem_tgt = semantic_pred_tgt_ema.clone()

        if not geom_augment:
            # Apply interpolation for mixup
            img_tgt = F.interpolate(img_tgt, img_src.shape[-2:], mode="bilinear",
                                    align_corners=False)
            lbl_sem_tgt = F.interpolate(lbl_sem_tgt.unsqueeze(1), img_src.shape[-2:],
                                        mode="nearest").squeeze(1)
            lbl_instance_tgt = F.interpolate(lbl_instance_tgt.unsqueeze(1).float(),
                                             img_src.shape[-2:],
                                             mode="nearest").type(torch.int32).squeeze(1)

        cls_msk = Mixup._get_cls_msk(lbl_sem_src)

        # Convert back to boolean
        cls_msk = (cls_msk > 0).unsqueeze(1)

        # Reshape cls msk to mask out all channels along dim 1
        cls_msk_rgb = cls_msk.repeat(1, img_src.shape[1], 1, 1)

        # Get the mixture of images
        rgb_mixup = img_src.float().clone()
        rgb_mixup[cls_msk_rgb] = img_tgt.float()[cls_msk_rgb]

        # Get the corresponding semantic image
        lbl_sem_mixup = lbl_sem_src.clone()
        # Attention: We do not take the cls_msk but use the semantic map to recompute the mask as
        # there might be pixels that are set to 255 in lbl_sem that should be considered for ema
        # plabels aswell!!
        # sem_msk = (lbl_sem_mixup == 255)
        lbl_sem_mixup[cls_msk] = lbl_sem_tgt.unsqueeze(1)[cls_msk]
        lbl_sem_mixup = lbl_sem_mixup.squeeze(1)

        # Get the corresponding instance image
        lbl_inst_mixup = lbl_instance_src.clone()
        lbl_inst_mixup[cls_msk] = lbl_instance_tgt.unsqueeze(1)[cls_msk].type(torch.int32)
        lbl_inst_mixup = lbl_inst_mixup.squeeze(1)

        if get_off_cen:
            # Get final offset and center labels for mixup image
            lbl_center_mixup, lbl_offset_mixup = Mixup.get_off_cen(lbl_inst_mixup)
            return rgb_mixup, lbl_sem_mixup, lbl_inst_mixup, lbl_center_mixup, lbl_offset_mixup
        return rgb_mixup, lbl_sem_mixup, lbl_inst_mixup

    # This is a naive implementation with image resizing

    @staticmethod
    def cut_mixup(img_src: Tensor, img_tgt: Tensor, lbl_instance_src: Tensor, lbl_sem_src: Tensor,
                  semantic_pred_tgt_ema: Tensor, instance_pred_tgt_ema: Tensor, nof_vert_split: int,
                  nof_hor_split: int, nof_segments: int, tgt_is_replay: Tensor, get_off_cen=True):
        msk_src, msk_tgt = Mixup._get_cut_msk(nof_vert_split, nof_hor_split, img_tgt, img_src,
                                              semantic_pred_tgt_ema, tgt_is_replay, nof_segments)

        # Reshape masks to mask out all channels along dim 1
        msk_src_rgb = msk_src.repeat(1, img_src.shape[1], 1, 1)
        msk_tgt_rgb = msk_tgt.repeat(1, img_tgt.shape[1], 1, 1)
        msk_src_lbl = msk_src
        msk_tgt_lbl = msk_tgt

        # Get the mixture of images
        rgb_mixup = img_src.float().clone()
        rgb_mixup[msk_src_rgb] = img_tgt[msk_tgt_rgb].float()

        # Get the corresponding semantic image
        lbl_sem_mixup = lbl_sem_src.clone()
        lbl_sem_mixup[msk_src_lbl] = semantic_pred_tgt_ema.unsqueeze(1)[msk_tgt_lbl]
        lbl_sem_mixup = lbl_sem_mixup.squeeze(1)

        # Get the corresponding instance image
        lbl_inst_mixup = lbl_instance_src.clone()
        lbl_inst_mixup[msk_src_lbl] = instance_pred_tgt_ema.type(torch.int32).unsqueeze(1)[
            msk_tgt_lbl]

        if get_off_cen:
            # Get final offset and center labels for mixup image
            lbl_center_mixup, lbl_offset_mixup = Mixup.get_off_cen(lbl_inst_mixup)
            return rgb_mixup, lbl_sem_mixup, lbl_inst_mixup, lbl_center_mixup, lbl_offset_mixup
        return rgb_mixup, lbl_sem_mixup, lbl_inst_mixup

    @staticmethod
    def conf_instance_mixup(img_src: Tensor, img_tgt: Tensor, lbl_instance_src: Tensor,
                            lbl_sem_src: Tensor, semantic_pred_tgt_ema: Tensor,
                            center_pred_tgt_ema: Tensor, offset_pred_tgt_ema: Tensor,
                            instance_algo: InstanceSegAlgo, threshold: float, min_inst_size: int,
                            get_off_cen: bool = True):

        thing_msk_src = (lbl_instance_src != 0)

        # Get mask for pasting according to confidences
        msk_tgt_idx, msk_src_idx, instance_pred_tgt_ema = \
            Mixup.get_conf_mask(semantic_pred_tgt_ema, center_pred_tgt_ema, offset_pred_tgt_ema,
                                instance_algo, img_src.shape, threshold, min_inst_size,
                                thing_msk_src)

        rgb_mixup = img_src.float().clone()
        lbl_sem_mixup = lbl_sem_src.clone().squeeze(1)
        lbl_inst_mixup = lbl_instance_src.clone()

        for b in range(img_src.shape[0]):
            if msk_src_idx[b] is None:
                continue

            rgb_mixup[b, :, msk_src_idx[b][:, 0], msk_src_idx[b][:, 1]] = img_tgt[b, :,
                                                                          msk_tgt_idx[b][:, 0],
                                                                          msk_tgt_idx[b][:,
                                                                          1]].float()
            lbl_sem_mixup[b, msk_src_idx[b][:, 0], msk_src_idx[b][:, 1]] = semantic_pred_tgt_ema[
                b, msk_tgt_idx[b][:, 0], msk_tgt_idx[b][:, 1]]
            lbl_inst_mixup[b, :, msk_src_idx[b][:, 0], msk_src_idx[b][:, 1]] = \
                instance_pred_tgt_ema[b, msk_tgt_idx[b][:, 0], msk_tgt_idx[b][:, 1]].type(
                    torch.int32)

        # # Reshape masks for mixup
        # msk_src_rgb = msk_src.repeat(1, img_src.shape[1], 1, 1)
        # msk_tgt_rgb = msk_tgt.repeat(1, img_tgt.shape[1], 1, 1)
        # msk_src_lbl = msk_src
        # msk_tgt_lbl = msk_tgt
        #
        # # Create rgb mixup image
        # rgb_mixup = img_src.float().clone()
        # rgb_mixup[msk_src_rgb] = img_tgt.float()[msk_tgt_rgb]
        #
        # # Create semantic mixup image
        # lbl_sem_mixup = lbl_sem_src.clone()
        # lbl_sem_mixup[msk_src_lbl] = semantic_pred_tgt_ema.unsqueeze(1)[msk_tgt_lbl]
        # lbl_sem_mixup = lbl_sem_mixup.squeeze(1)
        #
        # # Create instance mixup image
        # lbl_inst_mixup = lbl_instance_src.clone()
        # lbl_inst_mixup[msk_src_lbl] = instance_pred_tgt_ema.type(torch.int32).unsqueeze(1)[
        #     msk_tgt_lbl]

        if get_off_cen:
            # Get final offset and center labels for mixup image
            lbl_center_mixup, lbl_offset_mixup = Mixup.get_off_cen(lbl_inst_mixup)
            return rgb_mixup, lbl_sem_mixup, lbl_inst_mixup, lbl_center_mixup, lbl_offset_mixup
        return rgb_mixup, lbl_sem_mixup, lbl_inst_mixup

    @staticmethod
    def do_mixup(mixup_strategy: str, mixup_data: Dict[str, Any], instance_algo: InstanceSegAlgo,
                 cfg_mixup: CN) -> Dict[str, Any]:
        img_src = mixup_data["rgb_src"][0].clone()
        img_tgt = mixup_data["rgb_tgt"][0].clone()
        semantic_src = mixup_data["semantic_src"].clone()
        semantic_pred_tgt_ema = mixup_data["semantic_pred_tgt_ema"].detach().clone()
        instance_src = mixup_data["instance_src"].clone()
        center_pred_tgt_ema = mixup_data["center_pred_tgt_ema"].detach().clone()
        offset_pred_tgt_ema = mixup_data["offset_pred_tgt_ema"].detach().clone()
        _, instance_pred_tgt_ema = instance_algo.panoptic_fusion(semantic_pred_tgt_ema,
                                                                 center_pred_tgt_ema,
                                                                 offset_pred_tgt_ema)

        # Convert from the target camera to the source camera respecting their intrinsics
        if cfg_mixup.general.geom_augment:
            cam_models_src, cam_models_tgt = [], []
            for i in range(img_src.shape[0]):
                cam_models_src.append(
                    CameraModel.from_tensor(img_src.shape[-1], img_src.shape[-2], mixup_data[
                        "camera_model_src"][i]))
                cam_models_tgt.append(
                    CameraModel.from_tensor(img_tgt.shape[-1], img_tgt.shape[-2], mixup_data[
                        "camera_model_tgt"][i]))

            img_tgt = Mixup.warp_c2c(cam_models_tgt, cam_models_src, img_tgt, img_src,
                                     interp_mode="bilinear", padding_mode="zeros")
            instance_pred_tgt_ema = Mixup.warp_c2c(cam_models_tgt, cam_models_src,
                                                   instance_pred_tgt_ema, instance_src,
                                                   interp_mode="nearest",
                                                   padding_mode="zeros").type(
                instance_pred_tgt_ema.dtype).squeeze(1)
            semantic_pred_tgt_ema = Mixup.warp_c2c(cam_models_tgt, cam_models_src,
                                                   semantic_pred_tgt_ema, semantic_src,
                                                   interp_mode="nearest",
                                                   padding_mode="border").type(
                semantic_pred_tgt_ema.dtype).squeeze(1)

        # center and offset predictions according to geometric augmentation
        center_pred_tgt_ema, offset_pred_tgt_ema = Mixup.get_off_cen(instance_pred_tgt_ema)

        if mixup_strategy == "class_mixup":
            # Class mixup, ToDo: The confidences are not used yet...
            rgb_mixup, lbl_sem_mixup, lbl_inst_mixup, lbl_center_mixup, lbl_offset_mixup = \
                Mixup.class_mixup(img_src, img_tgt, instance_src, semantic_src,
                                  semantic_pred_tgt_ema, instance_pred_tgt_ema,
                                  cfg_mixup.general.geom_augment,
                                  cfg_mixup.class_mix.conf_thresh)
        elif mixup_strategy == "cut_mixup":
            # Cut mixup
            rgb_mixup, lbl_sem_mixup, lbl_inst_mixup, lbl_center_mixup, lbl_offset_mixup = \
                Mixup.cut_mixup(img_src, img_tgt, instance_src, semantic_src, semantic_pred_tgt_ema,
                                instance_pred_tgt_ema, cfg_mixup.cut_mix.nof_vert_splits,
                                cfg_mixup.cut_mix.nof_hor_splits, cfg_mixup.cut_mix.nof_segments,
                                mixup_data['tgt_is_replay'])
        elif mixup_strategy == "conf_instance_mixup":
            # Conf instance mixup
            rgb_mixup, lbl_sem_mixup, lbl_inst_mixup, lbl_center_mixup, lbl_offset_mixup = \
                Mixup.conf_instance_mixup(img_src, img_tgt, instance_src, semantic_src,
                                          semantic_pred_tgt_ema, center_pred_tgt_ema,
                                          offset_pred_tgt_ema, instance_algo,
                                          cfg_mixup.conf_instance_mix.conf_thresh,
                                          cfg_mixup.conf_instance_mix.min_inst_size)
        # elif mixup_strategy == "instance_mixup":
        #     # Instance mixup
        #     rgb_mixup, lbl_sem_mixup, lbl_inst_mixup, msk_mixup, lbl_inst_src = \
        #         Mixup.instance_mixup(instance_src.clone(), semantic_src.clone(),
        #                                    rgb_src.clone(), rgb_tgt.clone())
        # elif mixup_strategy == "geom_instance_mixup":
        #     # Geometric instance mixup, i.e. with cam2cam warping and geometric augmentation
        #     rgb_mixup, lbl_sem_mixup, lbl_inst_mixup = Mixup.mixer.geom_instance_mixup(
        #         instance_src.clone(), semantic_src.clone(), rgb_src.clone(),
        #         rgb_tgt.clone(), cam_model_src, cam_model_tgt)
        else:
            raise NotImplementedError("The requested mixup strategy is not implemented yet")

        output = {
            "rgb": {0: rgb_mixup},
            "camera_model": mixup_data["camera_model_src"],
            "semantic": lbl_sem_mixup,
            "center": lbl_center_mixup,
            "offset": lbl_offset_mixup,
            "instance": lbl_inst_mixup.squeeze(1),
        }
        return output

    # def check_occlusion(self, lbl_inst_i: torch.Tensor, lbl_inst_orig: torch.Tensor):
    #     lbl_inst_dil_i = M.dilation(lbl_inst_i, torch.ones(2,2))
    #     msk = lbl_inst_dil_i > 0
    #     id_vals = torch.unique(lbl_inst_orig[msk])
    #
    #     len_ids = 0
    #     len_ids += 1 for i in id_vals if i != 0

    # def resolve_occlusions(self, lbl_inst: torch.Tensor):
    #     return lbl_inst.clone()
    # for instance_id_i in torch.unique(lbl_inst_src2tgt).tolist():
    #     if instance_id_i != 0:
    # res_lbl_inst = lbl_inst.clone()
    # # ToDo: Implement this
    # return res_lbl_inst

    def instance_mixup(self, lbl_inst: torch.Tensor, lbl_sem: torch.Tensor, img_src: torch.Tensor,
                       img_tgt: torch.Tensor, geom_warping: Optional[bool] = False):
        img_src = F.interpolate(img_src, img_tgt.shape[-2:], mode="bilinear", align_corners=False)
        lbl_inst = F.interpolate(lbl_inst.unsqueeze(0), img_tgt.shape[-2:],
                                 mode="nearest").squeeze(0).type(torch.int32)
        lbl_sem = F.interpolate(lbl_sem.unsqueeze(0), img_tgt.shape[-2:],
                                mode="nearest").squeeze(0)

        inst_msk = self._get_inst_msk(lbl_inst)
        # Convert back to boolean
        inst_msk = inst_msk > 0

        # Reshape inst msk to mask out all channels along dim 1
        inst_msk = inst_msk.unsqueeze(1).repeat(1, img_src.shape[1], 1, 1)

        # Get the mixture of images
        out_rgb = img_tgt.clone()
        out_rgb[inst_msk] = img_src[inst_msk]

        # Get the corresponding semantic labels
        out_sem = lbl_sem.clone()
        out_sem[~inst_msk[:, 0, :, :]] = 255

        # Get the corresponding instance labels
        out_inst = lbl_inst.clone()
        out_inst[~inst_msk[:, 0, :, :]] = 0

        # Attention: We do not take the cls_msk but use the semantic map to recompute the mask as
        # there might be pixels that are set to 255 in lbl_sem that should be considered for ema
        # plabels aswell!!
        out_msk = (out_sem == 255)

        return out_rgb, out_sem, out_inst, out_msk, lbl_inst

    @staticmethod
    def get_conf_mask(semantic_pred_ema: Tensor, center_ema_plabel_det: Tensor,
                      offset_ema_plabel_det: Tensor, instance_algo: InstanceSegAlgo,
                      src_shape: ArrayLike, threshold: float, min_inst_size: int,
                      thing_msk_src: Tensor):
        _, instance_plabel_ema = instance_algo.panoptic_fusion(semantic_pred_ema,
                                                               center_ema_plabel_det,
                                                               offset_ema_plabel_det,
                                                               threshold_center=threshold)
        instance_plabel_ema = instance_plabel_ema.squeeze(1)

        msk_tgt = torch.zeros_like(instance_plabel_ema).type(torch.int64)
        msk_src = torch.zeros((instance_plabel_ema.shape[0], src_shape[-2], src_shape[-1]),
                              dtype=msk_tgt.dtype, device=instance_plabel_ema.device)

        msk_tgt_idx = []
        msk_src_idx = []

        for b in range(instance_plabel_ema.shape[0]):
            offsets = {}
            y_positions = {}

            instance_id_mask = torch.zeros((src_shape[-2], src_shape[-1]),
                                           dtype=instance_plabel_ema.dtype,
                                           device=instance_plabel_ema.device)

            thing_msk = thing_msk_src[b].clone().squeeze()

            msk_tgt_idx.append([])
            msk_src_idx.append([])

            for conf_id in instance_plabel_ema[b].unique():
                conf_id = conf_id.item()
                if conf_id == 0:
                    continue  # Not a thing class
                inst_size = (instance_plabel_ema[b] == conf_id).sum()
                if inst_size < min_inst_size:
                    continue  # Skip small instances

                center = torch.round(
                    (instance_plabel_ema[0] == conf_id).nonzero().float().mean(0)).int()

                msk_tgt[b][instance_plabel_ema[b] == conf_id] = 2  # Will be set to 1 later
                idxs_tgt = (instance_plabel_ema[b] == conf_id).nonzero()

                # Move to the maximum distance position here
                dist_map = distance_transform(
                    thing_msk.type(torch.float32).unsqueeze(0).unsqueeze(0)).squeeze()
                dist_row = dist_map[center[0], :]
                idx_max = torch.argmax(dist_row)
                offset_x = idx_max - center[1]

                idxs_src = idxs_tgt.clone()
                idxs_src[:, 1] += offset_x

                # In the source image
                min_x = torch.min(idxs_src[:, 1])
                max_x = torch.max(idxs_src[:, 1])
                if min_x < 0:
                    idxs_src[:, 1] -= min_x
                    offset_x -= min_x
                if max_x > instance_plabel_ema.shape[-1] - 1:
                    idxs_src[:, 1] -= max_x - (instance_plabel_ema.shape[-1] - 1)
                    offset_x -= max_x - (instance_plabel_ema.shape[-1] - 1)

                offsets[conf_id] = offset_x
                y_positions[conf_id] = center[0]

                msk_src[b, idxs_src[:, 0], idxs_src[:, 1]] += 1

                if torch.any(msk_src[b] > 1):
                    conflicting_instance_ids = instance_id_mask[msk_src[b] > 1].unique()

                    for conflicting_instance_id in conflicting_instance_ids:
                        conflicting_instance_id = conflicting_instance_id.item()

                        conflicting_indices = torch.logical_and(
                            msk_src[b] > 1, instance_id_mask == conflicting_instance_id).nonzero()

                        # The new instance is further away from the camera
                        # if center[0] > y_positions[conflicting_instance_id]:
                        # Thus, crop the new instance
                        conflicting_indices[:, 1] -= offset_x
                        msk_tgt[b][conflicting_indices[:, 0], conflicting_indices[:, 1]] = 0

                        surviving_indices = torch.logical_and(
                            msk_src[b] == 1, instance_id_mask == 0).nonzero()
                        instance_id_mask[
                            surviving_indices[:, 0], surviving_indices[:, 1]] = conf_id

                        # The new instance is closer to the camera
                        # else:
                        #     # Thus, crop the conflicting instance
                        #     conflicting_indices[:, 1] -= offsets[conflicting_instance_id]
                        #     msk_tgt[b][conflicting_indices[:, 0], conflicting_indices[:, 1]] = 0
                        #
                        #     instance_id_mask[idxs_src[:, 0], idxs_src[:, 1]] = conf_id

                else:
                    instance_id_mask[idxs_src[:, 0], idxs_src[:, 1]] = conf_id

                msk_src_idx[b].append((instance_id_mask == conf_id).nonzero())
                msk_tgt_idx[b].append((msk_tgt[b] == 2).nonzero())

                msk_src = torch.clamp(msk_src, 0, 1)
                msk_tgt = torch.clamp(msk_tgt, 0, 1)

                # Update the thing mask to recompute the distance map
                thing_msk[idxs_src[:, 0], idxs_src[:, 1]] = 1

            if msk_src_idx[b]:
                msk_src_idx[b] = torch.cat(msk_src_idx[b])
                msk_tgt_idx[b] = torch.cat(msk_tgt_idx[b])
            else:
                msk_src_idx[b] = None
                msk_tgt_idx[b] = None

        return msk_tgt_idx, msk_src_idx, instance_plabel_ema

    # @staticmethod
    # def paste_instance(msk, instance_plabel, lbl_instance: torch.Tensor, lbl_sem: torch.Tensor,
    #                    plbl_instance: torch.Tensor, plbl_sem: torch.Tensor,
    #                    img_src: torch.Tensor, img_tgt: torch.Tensor):

    # def geom_instance_mixup(self, lbl_inst_src: torch.Tensor, lbl_sem_src: torch.Tensor,
    #                         img_src: torch.Tensor, img_tgt: torch.Tensor,
    #                         cam_model_src: List[CameraModel],
    #                         cam_model_tgt: List[CameraModel],
    #                         geom_warping: Optional[bool] = False):
    #
    #     # 1.) Resolve occlusions, i.e. create a new tensor with all instances separate
    #     res_lbl_inst_src = lbl_inst_src
    #     # ToDo: Take this in again
    #     # res_lbl_inst_src = self.resolve_occlusions(lbl_inst_src)
    #
    #     # 2.) Warp all tensors in advance to the target image
    #     lbl_inst_src2tgt = self.warp_c2c(cam_model_src, cam_model_tgt, res_lbl_inst_src, img_tgt,
    #                                      interp_mode="nearest").type(torch.int32)
    #     lbl_sem_src2tgt = self.warp_c2c(cam_model_src, cam_model_tgt, lbl_sem_src, img_tgt,
    #                                     interp_mode="nearest").type(torch.uint8)
    #     img_src2tgt = self.warp_c2c(cam_model_src, cam_model_tgt, img_src, img_tgt,
    #                                 interp_mode="bilinear", padding_mode="zeros")
    #
    #     # Mask out by the valid region
    #
    #     # 3.) Preallocate mixup images for later pasting process
    #     lbl_inst_mixup = torch.zeros_like(lbl_inst_src2tgt)
    #     lbl_sem_mixup = torch.ones_like(lbl_sem_src2tgt) * 255
    #     img_mixup = img_tgt.clone()
    #
    #     # 4.) Go through each instance, geometrically augment and paste into mixup image in the
    #     # target domain
    #     for instance_id_i in torch.unique(lbl_inst_src2tgt).tolist():
    #         if instance_id_i != 0:
    #             instance_i = torch.zeros_like(lbl_inst_src2tgt)
    #             instance_i[lbl_inst_src2tgt == instance_id_i] = lbl_inst_src2tgt[lbl_inst_src2tgt ==
    #                                                                              instance_id_i]
    #
    #             if geom_warping:
    #                 instance_aug_i, lbl_sem_aug, img_aug, msk = \
    #                     self.process_instance(instance_i, lbl_sem_src2tgt, img_src2tgt,
    #                                           cam_model_tgt)
    #             else:
    #                 instance_aug_i, lbl_sem_aug, img_aug, msk = \
    #                     self.process_instance(instance_i, lbl_sem_src2tgt, img_src2tgt)
    #
    #             # Paste warped and geom. augmented parts from the src2tgt image into the mixup image
    #             lbl_inst_mixup[msk] = instance_aug_i[msk]
    #             lbl_sem_mixup[msk] = lbl_sem_aug[msk]
    #             img_mixup[msk.repeat(1, img_mixup.shape[1], 1, 1)] = img_aug[
    #                 msk.repeat(1, img_mixup.shape[1], 1, 1)].float()
    #
    #     return img_mixup, lbl_sem_mixup.squeeze(1), lbl_inst_mixup.squeeze(1)

    def process_instance(self, lbl_inst_i: torch.Tensor, lbl_sem: torch.Tensor, img: torch.Tensor,
                         cam_model: Optional[List[CameraModel]] = None):

        # 1. Apply geometric augmentation if cam_model_tgt is set
        if cam_model is not None:
            lbl_inst_aug_i, lbl_sem_aug, img_aug = self.geometric_augmentation(lbl_inst_i, lbl_sem,
                                                                               img, cam_model)
        else:
            lbl_inst_aug_i, lbl_sem_aug, img_aug = lbl_inst_i, lbl_sem, img

        # Get the mask to carve out the corresponding part from the src2tgt image lateron
        msk = lbl_inst_aug_i != 0
        return lbl_inst_aug_i, lbl_sem_aug, img_aug, msk

    # @staticmethod
    # def geometric_augmentation(lbl_inst, lbl_sem, img, cam_model):
    #     NotImplementedError("not implemented yet")
    #     return lbl_inst, lbl_sem, img

    @staticmethod
    def _gen_rnd_inst(lbl_inst: torch.Tensor):
        all_inst = torch.unique(lbl_inst)
        num_inst = all_inst.shape[0]
        return (all_inst[torch.Tensor(
            np.random.choice(num_inst, int((num_inst + num_inst % 2) / 2), replace=False)).long()])

    @staticmethod
    def _gen_instance_msk(lbl_inst: torch.Tensor, inst_ids: torch.Tensor):
        lbl_inst_cast, inst_cast = torch.broadcast_tensors(lbl_inst.unsqueeze(0),
                                                           inst_ids.unsqueeze(1).unsqueeze(2))
        return torch.eq(lbl_inst_cast, inst_cast).sum(1)

    def _get_inst_msk(self, lbl_inst: torch.Tensor):
        rnd_inst = self._gen_rnd_inst(lbl_inst)
        inst_msk = self._gen_class_msk(lbl_inst, rnd_inst)
        return inst_msk

    # Fixme: Not sure / 255.0 will be needed in the main code. Depends on whether the image is
    #  normalized or not
    def style_transfer(self, img_ref: torch.Tensor, img_tgt: torch.Tensor, mode: str):
        img_tgt_pil = T.ToPILImage()(img_tgt.squeeze(0))
        img_ref_pil = T.ToPILImage()(img_ref.squeeze(0))
        img_ref_resized = img_ref_pil.resize(img_tgt_pil.size)
        out = transfer_histogram_style(img_tgt_pil, img_ref_resized, mode)
        out = (torch.from_numpy(out) / 255.0).permute(2, 0, 1).unsqueeze(0)
        return out
