from typing import Dict, Optional, Tuple, Union

import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn

from eval import PanopticEvaluator
from misc.post_processing_panoptic import get_panoptic_segmentation
from models import InstanceHead


class CenterLoss:

    def __init__(self):
        self.mse_loss = nn.MSELoss(reduction="none")

    def __call__(self, prediction: Tensor, target: Tensor, pixel_weights: ArrayLike) -> Tensor:
        """
        Parameters
        ----------
        pixel_weights : torch.Tensor
            Ignore region with 0 is ignore and 1 is consider
        """
        loss = self.mse_loss(prediction, target)
        return loss.mean()


class OffsetLoss:

    def __init__(self):
        self.l1_loss = nn.L1Loss(reduction="none")

    def __call__(self, prediction: Tensor, target: Tensor, pixel_weights: ArrayLike) -> Tensor:
        """
        Parameters
        ----------
        pixel_weights : torch.Tensor
            Ignore region with 0 is ignore and 1 is consider
        """
        loss = self.l1_loss(prediction, target)
        return loss.mean()


class BinaryMaskLoss:

    def __init__(self, ignore_index: int = 255):
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        loss = self.ce_loss(prediction, target.long())
        return loss.mean()


# -------------------------------------------------------- #


class InstanceSegAlgo:
    """
    Parameters
    ----------
    thing_list :
        These IDs represent object instances ("things").
    """

    def __init__(
            self,
            center_loss: CenterLoss,
            offset_loss: OffsetLoss,
            evaluator: PanopticEvaluator,
            binary_mask_loss: Optional[BinaryMaskLoss] = None,
    ):
        self.center_loss = center_loss
        self.offset_loss = offset_loss
        self.binary_mask_loss = binary_mask_loss
        self.evaluator = evaluator

    def training(
            self,
            feats: Tensor,
            center: Tensor,
            offset: Tensor,
            center_weights: Tensor,
            offset_weights: Tensor,
            thing_mask: Tensor,
            instance_head: InstanceHead
    ):
        """
        Parameters
        ----------
        feats : torch.Tensor
            Features from the multi-task encoder used as input to depth, segmentation, etc.
        center : torch.Tensor
            Ground truth heatmap of instance centers
        offset : torch.Tensor
            Ground truth offset (x,y) for each pixel to the closest instance center
        center_weights : torch.Tensor
            Pixel-wise loss weights for center loss
        offset_weights : torch.Tensor
            Pixel-wise loss weights for offset loss
        instance_head: InstanceHead
            Decoder for predicting instance centers and offsets
        """
        center_pred, offset_pred, thing_mask_logits = instance_head(feats)

        # Compute losses
        center_loss = self.center_loss(center_pred, center, center_weights)
        offset_loss = self.offset_loss(offset_pred, offset, offset_weights)

        if thing_mask_logits is not None:
            thing_mask_loss = self.binary_mask_loss(thing_mask_logits, thing_mask.squeeze(1))
            thing_mask_pred = torch.argmax(thing_mask_logits, dim=1).type(torch.uint8)
        else:
            thing_mask_loss, thing_mask_pred = None, None

        return center_loss, offset_loss, center_pred, offset_pred, thing_mask_loss, thing_mask_pred

    def inference(
            self,
            feats: Tensor,
            instance_head: InstanceHead,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Parameters
        ----------
        feats : torch.Tensor
            Features from the multi-task encoder used as input to depth, segmentation, etc.
        instance_head: InstanceHead
            Decoder for predicting instance centers and offsets
        """
        center_pred, offset_pred, thing_mask_logits = instance_head(feats)

        if thing_mask_logits is not None:
            thing_mask_pred = torch.argmax(thing_mask_logits, dim=1).type(torch.uint8)
        else:
            thing_mask_pred = None

        return center_pred, offset_pred, thing_mask_pred

    def evaluation(self):
        raise NotImplementedError

    def adaptation(
            self,
            feats: Dict[str, Tuple[Tensor, Tensor, Tensor]],
            center: Dict[str, Tensor],
            offset: Dict[str, Tensor],
            center_weights: Dict[str, Tensor],
            offset_weights: Dict[str, Tensor],
            thing_mask: Dict[str, Tensor],
            instance_head: InstanceHead,
            center_ema: Optional[Tensor] = None,
            offset_ema: Optional[Tensor] = None,
    ) -> Union[Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor], Tuple[
        Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor, Dict[str, Tensor], Tensor]]:
        # ---------------------------------------
        center_pred, offset_pred, thing_mask_pred = {}, {}, {}
        center_pred["target"], offset_pred["target"], thing_mask_pred["target"] = self.inference(
            feats["target"][0], instance_head)

        # ---------------------------------------
        # EMA consistency loss
        if center_ema is not None:
            center_ema_cons_loss = -torch.sigmoid(center_pred["target"]) * torch.log(
                torch.sigmoid(center_ema) + 1e-10)
            center_ema_cons_loss = center_ema_cons_loss.mean()
        else:
            center_ema_cons_loss = None
        if offset_ema is not None:
            offset_ema_cons_loss = -torch.sigmoid(offset_pred["target"]) * torch.log(
                torch.sigmoid(offset_ema) + 1e-10)
            offset_ema_cons_loss = offset_ema_cons_loss.mean()
        else:
            offset_ema_cons_loss = None

        # ---------------------------------------
        # Normal supervised loss
        if "source" in feats:
            supervised_center_loss, supervised_offset_loss, _, _, supervised_thing_mask_loss, _, = \
                self.training(
                    feats["source"][0], center["source"], offset["source"],
                    center_weights["source"], offset_weights["source"], thing_mask["source"],
                    instance_head)
        else:
            supervised_center_loss, supervised_offset_loss, supervised_thing_mask_loss = \
                None, None, None

        # ---------------------------------------
        # Collect return values
        center_losses = {
            "source": supervised_center_loss,
            "ema": center_ema_cons_loss,
        }
        offset_losses = {
            "source": supervised_offset_loss,
            "ema": offset_ema_cons_loss,
        }
        thing_mask_losses = {
            "source": supervised_thing_mask_loss,
        }

        if instance_head.use_thing_mask:
            return center_losses, offset_losses, center_pred["target"], offset_pred[
                "target"], thing_mask_losses, thing_mask_pred["target"]

        return center_losses, offset_losses, center_pred["target"], offset_pred["target"]

    def panoptic_fusion(
            self,
            semantic: Tensor,
            center: Tensor,
            offset: Tensor,
            return_center: bool = False,
            threshold_center: float = None,
            thing_mask: Optional[Tensor] = None,
            do_merge_semantic_and_instance=True,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Note a change in the void label:
        - semantic map: 255
        - panoptic map: -1
        """

        batch_size = semantic.shape[0]
        if do_merge_semantic_and_instance:
            # Int16 since the largest expected number is 18 * 1000 < 32767
            panoptic = torch.empty_like(semantic, dtype=torch.int16)
        else:
            panoptic = None
        instance = torch.empty_like(semantic)

        center_pts = []

        for i in range(batch_size):
            thing_list = self.evaluator.thing_list
            label_divisor = 1000  # pan_ID = sem_class_id * label_divisor + inst_id
            stuff_area = 0
            void_label = 255
            threshold = .1 if threshold_center is None else threshold_center
            nms_kernel = 7
            top_k = 200
            foreground_mask = thing_mask[i].unsqueeze(0) if thing_mask is not None else None

            semantic_ = semantic[i, :].unsqueeze(0)
            center_ = center[i, :].unsqueeze(0)
            offset_ = offset[i, :].unsqueeze(0)

            pan, cnt, inst = get_panoptic_segmentation(semantic_, center_, offset_, thing_list,
                                                       label_divisor, stuff_area, void_label,
                                                       threshold, nms_kernel, top_k,
                                                       foreground_mask,
                                                       do_merge_semantic_and_instance)
            center_pts.append(cnt)
            if do_merge_semantic_and_instance:
                panoptic[i] = pan
            instance[i] = inst
        if return_center:
            return panoptic, instance, center_pts
        return panoptic, instance
