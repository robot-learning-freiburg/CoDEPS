# pylint: disable=condition-evals-to-constant, unused-argument, unused-import

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from eval import SemanticEvaluator
from misc import CameraModel, ImageWarper
from misc.post_processing_panoptic import find_instance_center, group_pixels
from models import SemanticHead


class SemanticLoss:
    """Hard pixel mining with cross entropy loss, for semantic segmentation.
    Following DeepLab Cross Entropy loss
    https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/loss/criterion.py
    """

    def __init__(self,
                 device,
                 ignore_index: int = 255,
                 ignore_labels: Optional[List] = None,
                 top_k_percent_pixels: float = 1.0,
                 class_weights: Optional[Tuple[float, ...]] = None):
        if not 0. < top_k_percent_pixels <= 1.0:
            raise ValueError('top_k_percent_pixels must be within (0, 1]')
        self.device = device
        self.ignore_labels = ignore_labels
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_index = ignore_index
        weight = None
        if class_weights is not None:
            if ignore_labels is None:
                weight = torch.tensor(class_weights, device=device)
            else:
                weight = torch.tensor(
                    [w for label, w in enumerate(class_weights) if label not in ignore_labels],
                    device=device)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight,
                                           ignore_index=ignore_index,
                                           reduction='none')

    def __call__(self, prediction_softmax: Tensor, target: Tensor, pixel_weights: Tensor,
                 return_per_pixel: bool = False) -> Tensor:
        if return_per_pixel:
            assert self.top_k_percent_pixels == 1.0, 'top_k must be 1.0 for return_per_pixel = True'

        if self.ignore_labels is not None:
            for ignore_label in self.ignore_labels:
                target[target == ignore_label] = self.ignore_index
            preserved_labels = [label for label in range(prediction_softmax.shape[1]) if
                                label not in self.ignore_labels]
            prediction_softmax = prediction_softmax[:, preserved_labels, ...]

        loss = self.ce_loss(prediction_softmax, target.long()) * pixel_weights

        if self.top_k_percent_pixels < 1.0:
            loss = loss.contiguous().view(-1)
            # Hard pixel mining
            top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
            loss, _ = torch.topk(loss, top_k_pixels)

        if return_per_pixel:
            return loss
        return loss.mean()


class SemanticConsistencyLoss:
    def __init__(self,
                 device,
                 ref_img_width: int,
                 ref_img_height: int,
                 ignore_index: int = 255,
                 ignore_labels: Optional[List] = None,
                 class_weights: Optional[Tuple[float, ...]] = None):

        self.ignore_labels = ignore_labels
        self.image_warper = ImageWarper(ref_img_width, ref_img_height, device)
        self.semantic_loss = SemanticLoss(device, ignore_index, ignore_labels,
                                          class_weights=class_weights)

    def _compute_loss(self, prediction_softmax: Tensor, target: Tensor,
                      pred_rgb_img: Optional[Tensor] = None,
                      target_rgb_img: Optional[Tensor] = None) -> Tensor:
        entropy = -torch.sum((prediction_softmax * torch.log(prediction_softmax + 1e-10)), dim=1)
        pixel_weights = (entropy.max() - entropy) / entropy.max()
        loss = self.semantic_loss(prediction_softmax, target, pixel_weights, return_per_pixel=True)

        # Fixme: This may need some appropriate weighting as the exp can weight down the loss a lot
        #  such that it has a small impact
        if pred_rgb_img is not None and target_rgb_img is not None:
            # Inspired by EdgeAwareSmoothnessLoss
            loss *= torch.exp(-torch.mean(torch.abs(target_rgb_img - pred_rgb_img), 1))

        return loss.unsqueeze(1)

    def __call__(
            self,
            camera_models: List[CameraModel],
            preds_logits: Tuple[Tensor, Tensor, Tensor],
            images: Tuple[Tensor, Tensor, Tensor],
            depth_map: Tensor,
            poses: Tuple[Tensor, Tensor],
            pixel_weights: Tensor,
            object_motion_maps: Optional[Tuple[Tensor, Tensor]] = None, ) -> Tensor:
        assert len(camera_models) == preds_logits[0].shape[0], \
            'Batch size of camera model does not match batch size of predicted logits'
        # ToDo: Do a filtering (threshold based) based on entropy before translating adjacent
        #  semantic images to target images to serve as pseudo gt
        sem_preds = [torch.argmax(pred_logits_i, dim=1) for pred_logits_i in preds_logits]

        recon_cel_loss = []
        for i, (sem_pred_i, img_i, poses_i) in enumerate(zip(sem_preds[1:], images[1:], poses)):
            sem_pred_i = sem_pred_i.unsqueeze(1).float()
            if object_motion_maps is None:
                warped_sem_pred_i = self.image_warper(camera_models, sem_pred_i,
                                                      depth_map, poses_i, interp_mode='nearest')
                warped_rgb_i = self.image_warper(camera_models, img_i, depth_map, poses_i)
            else:
                warped_sem_pred_i = self.image_warper(camera_models, sem_pred_i, depth_map,
                                                      poses_i, interp_mode='nearest',
                                                      object_motion_map=object_motion_maps[i])
                warped_rgb_i = self.image_warper(camera_models, img_i, depth_map, poses_i,
                                                 object_motion_map=object_motion_maps[i])

            recon_cel_loss.append(self._compute_loss(preds_logits[0], warped_sem_pred_i.squeeze(1),
                                                     warped_rgb_i, images[0]))
        recon_cel_loss = torch.cat(recon_cel_loss, 1)

        # Auto masking for semantics as well
        identity_losses = []
        for sem_pred_i in sem_preds[1:]:
            identity_losses.append(self._compute_loss(preds_logits[0], sem_pred_i))
        identity_losses = torch.cat(identity_losses, 1)
        # Add random numbers to break ties
        identity_losses += torch.randn(identity_losses.shape, device=depth_map.device) * .00001

        combined_losses = torch.cat((recon_cel_loss, identity_losses), dim=1)
        loss_per_pixel, _ = torch.min(combined_losses, dim=1)

        loss = loss_per_pixel.mean()
        return loss


# -------------------------------------------------------- #


class SemanticSegAlgo:

    def __init__(
            self,
            semantic_loss: SemanticLoss,
            evaluator: SemanticEvaluator,
            semantic_consistency_loss: Optional[SemanticConsistencyLoss] = None,
    ):
        self.semantic_loss = semantic_loss
        self.semantic_consistency_loss = semantic_consistency_loss
        self.evaluator = evaluator

    def training(
            self,
            feats: Tensor,
            semantic_head: SemanticHead,
            semantic_gt: Tensor,
            semantic_weights: Tensor,
            ignore_classes: Optional[List] = None,
            semantic_gt_eval: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        semantic_logits = semantic_head(feats)

        if ignore_classes is not None:
            semantic_logits_ignored = semantic_logits.detach().clone()
            for ignore_class in ignore_classes:
                semantic_logits_ignored[:, ignore_class, :, :] = -float('inf')
            semantic_pred = torch.argmax(semantic_logits_ignored, dim=1).type(torch.uint8)
        else:
            semantic_pred = torch.argmax(semantic_logits, dim=1).type(torch.uint8)
        if semantic_gt_eval is not None:
            confusion_matrix = self.evaluator.compute_confusion_matrix(semantic_pred, semantic_gt_eval)
        else:
            confusion_matrix = self.evaluator.compute_confusion_matrix(semantic_pred, semantic_gt)

        semantic_loss = self.semantic_loss(semantic_logits, semantic_gt, semantic_weights)

        return semantic_loss, confusion_matrix, semantic_pred

    def inference(self, feats: Tensor, semantic_head: SemanticHead) -> Tuple[Tensor, Tensor]:
        semantic_logits = semantic_head(feats)
        semantic_pred = torch.argmax(semantic_logits, dim=1).type(torch.uint8)
        return semantic_pred, semantic_logits

    def evaluation(self, feats: Tensor, semantic_head: SemanticHead, semantic_gt: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        semantic_pred, semantic_logits = self.inference(feats, semantic_head)
        confusion_matrix = self.evaluator.compute_confusion_matrix(semantic_pred, semantic_gt)
        return confusion_matrix, semantic_pred, semantic_logits

    def adaptation(
            self,
            feats: Dict[str, Tuple[Tensor, Tensor, Tensor]],
            semantic_head: SemanticHead,
            semantic_gt: Dict[str, Tensor],
            semantic_weights: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        # ---------------------------------------
        semantic_logits = {
            'target': semantic_head(feats['target'][0])
        }

        # ---------------------------------------
        pred = {key: torch.argmax(value, dim=1) for key, value in semantic_logits.items()}

        # ---------------------------------------
        # Semantic Mixup Losses
        # ToDo: Use confidences to determine which pseudolabels from ema model predictions we
        #  want to use (yep)
        mixup_semantic_losses = {}
        for key, feat_mixup in feats.items():
            if key.endswith('mixup'):
                semantic_logits_pred_mixup = semantic_head(feat_mixup[0])
                mixup_semantic_losses[key] = self.semantic_loss(semantic_logits_pred_mixup,
                                                                semantic_gt[key],
                                                                torch.ones_like(semantic_gt[key]))

        # ---------------------------------------
        # Normal supervised loss
        if 'source' in feats:
            semantic_logits['source'] = semantic_head(feats['source'][0])
            supervised_loss = self.semantic_loss(semantic_logits['source'], semantic_gt['source'],
                                                 semantic_weights['source'])
        else:
            supervised_loss = None

        # ---------------------------------------
        # Collect return values
        semantic_losses = {
            'source': supervised_loss,
        }
        semantic_losses.update(mixup_semantic_losses)
        if semantic_gt['target'] is not None:
            confusion_matrix = self.evaluator.compute_confusion_matrix(pred['target'],
                                                                       semantic_gt['target'])
        else:
            confusion_matrix = None
        semantic_pred = pred['target']

        return semantic_losses, confusion_matrix, semantic_pred
