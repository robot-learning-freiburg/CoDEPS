from typing import Dict, Tuple

import torch
from torch import Tensor


class DepthEvaluator:
    """
    Evaluate depth prediction
    """

    def __init__(self, use_gt_scale: bool, depth_ranges: Tuple[float, float],
                 use_garg_crop: bool = False):
        """
        Args:
        """
        self.use_gt_scale = use_gt_scale
        self.depth_ranges = depth_ranges
        self.use_garg_crop = use_garg_crop

    def compute_depth_metrics(self, depth_gt: Tensor, depth_pred: Tensor) -> Dict[str, Tensor]:
        """Compute depth metrics, to allow monitoring during training
        Adapted from: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        if len(depth_gt.shape) == 3:
            depth_gt = depth_gt.unsqueeze(1)  # B, H, W -> B, 1, H, W
        min_depth = self.depth_ranges[0]
        max_depth = self.depth_ranges[1]

        depth_pred = depth_pred.detach()
        # depth_pred = torch.clamp(depth_pred, min_depth, max_depth)

        mask = depth_gt > 0  # Mask VOID data

        # garg/eigen crop
        if self.use_garg_crop:
            _, _, gt_height, gt_width = depth_gt.shape
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :,
            int(0.4080 * gt_height):int(0.9891 * gt_height),
            int(0.0354 * gt_width):int(0.9638 * gt_width)] = 1
            mask *= crop_mask

        batch_size = depth_gt.shape[0]
        depth_stats = {}
        for b in range(batch_size):
            gt, pred = depth_gt[b][mask[b]], depth_pred[b][mask[b]]

            # Normalize the depth predictions only if the training was fully unsupervised as the
            # scales cannot be estimated then
            if self.use_gt_scale:
                ratio = gt.median() / pred.median()
                pred *= ratio

            gt = torch.clamp(gt, min_depth, max_depth)
            pred = torch.clamp(pred, min_depth, max_depth)

            b_depth_stats = self._compute_depth_stats(gt, pred)

            for key, value in b_depth_stats.items():
                if key in depth_stats:
                    depth_stats[key] += value
                else:
                    depth_stats[key] = value
        for value in depth_stats.values():
            value /= batch_size

        return depth_stats

    def compute_depth_metrics_per_class(self, depth_gt: Tensor, depth_pred: Tensor,
                                        semantic_gt: Tensor) -> Dict[str, Tensor]:
        depth_gt = depth_gt.unsqueeze(1)  # B, H, W -> B, 1, H, W
        semantic_gt = semantic_gt.unsqueeze(1)  # B, H, W -> B, 1, H, W

        min_depth = self.depth_ranges[0]
        max_depth = self.depth_ranges[1]

        depth_stats = {}
        for c in torch.unique(semantic_gt):
            if c == 255:
                continue

            gt = depth_gt[semantic_gt == c]
            pred = depth_pred[semantic_gt == c]

            mask = gt > 0  # Mask VOID data
            if not mask.any():
                continue

            gt, pred = gt[mask], pred[mask]

            # Normalize the depth predictions only if the training was fully unsupervised as the
            # scales cannot be estimated then
            if self.use_gt_scale:
                ratio = gt.median() / pred.median()
                pred *= ratio

            gt = torch.clamp(gt, min_depth, max_depth)
            pred = torch.clamp(pred, min_depth, max_depth)

            c_depth_stats = self._compute_depth_stats(gt, pred)

            for key, value in c_depth_stats.items():
                depth_stats[f"{key}_c{c}"] = value
        return depth_stats

    @staticmethod
    def _compute_depth_stats(gt: Tensor, pred: Tensor) -> Dict[str, Tensor]:
        """Computation of error metrics between predicted and ground truth depths
        Source: https://github.com/nianticlabs/monodepth2/blob/master/layers.py
        """
        stats = {}

        thresh = torch.max((gt / pred), (pred / gt))
        stats["d_a1"] = (thresh < 1.25).float().mean()
        stats["d_a2"] = (thresh < 1.25 ** 2).float().mean()
        stats["d_a3"] = (thresh < 1.25 ** 3).float().mean()

        rmse = (gt - pred) ** 2
        stats["d_rmse"] = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
        stats["d_rmse_log"] = torch.sqrt(rmse_log.mean())

        stats["d_abs_rel"] = torch.mean(torch.abs(gt - pred) / gt)

        stats["d_sq_rel"] = torch.mean((gt - pred) ** 2 / gt)

        return stats
