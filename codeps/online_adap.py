import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from torch import Tensor, nn

from algos import DepthAlgo, InstanceSegAlgo, SemanticSegAlgo
from datasets.mixup import Mixup
from misc.camera_model import CameraModel
from models import (
    DepthHead,
    FlowHead,
    InstanceHead,
    PoseHead,
    ResnetEncoder,
    SemanticHead,
)


class CodepsNet(nn.Module):

    def __init__(self, cfg_mixup, backbone_po_depth: ResnetEncoder,
                 backbone_pose_sflow: Optional[ResnetEncoder], depth_head: Optional[DepthHead],
                 pose_head: Optional[PoseHead], flow_head: Optional[FlowHead],
                 semantic_head: Optional[SemanticHead], instance_head: Optional[InstanceHead],
                 depth_algo: Optional[DepthAlgo], semantic_algo: Optional[SemanticSegAlgo],
                 instance_algo: Optional[InstanceSegAlgo]):
        super().__init__()
        if depth_algo is not None:
            assert depth_head is not None
            assert pose_head is not None
            # Note: flow_head can be None
        if semantic_algo is not None:
            assert semantic_head is not None
        if instance_algo is not None:
            assert instance_head is not None

        self.cfg_mixup = cfg_mixup

        self.backbone_po_depth = backbone_po_depth
        self.backbone_pose_sflow = backbone_pose_sflow

        self.depth_head = depth_head
        self.pose_head = pose_head
        self.flow_head = flow_head
        self.semantic_head = semantic_head
        self.instance_head = instance_head

        self.depth_algo = depth_algo
        self.semantic_algo = semantic_algo
        self.instance_algo = instance_algo

    def forward(self,
                in_data: Dict[str, Tensor],
                mode: str = "infer",
                depth_offsets: Optional[List[int]] = None,
                do_panoptic_fusion: bool = False,
                do_class_wise_depth_stats: bool = False,
                sem_ignore_classes: Optional[List[int]] = None):
        assert mode in ["train", "eval", "infer", "adapt"], f"Unsupported mode: {mode}"
        if mode == "adapt":
            return self.adapt(in_data, depth_offsets, do_panoptic_fusion)
        # --------------------
        if mode == "train":
            assert depth_offsets is not None, "'depth_offsets' must be set in 'train' mode."
            # We assume depth_offsets = [0, -offset, +offset] with offset as a positive integer
            assert depth_offsets[0] == 0
            assert depth_offsets[1] < 0
            assert depth_offsets[1] == -depth_offsets[2]
        if do_panoptic_fusion:
            assert self.semantic_algo is not None
            assert self.instance_algo is not None

        # --------------------
        # In training mode, return the predictions, the statistics, and the loss(!)
        if mode == "train":
            # Get images for the window
            images_window = [in_data["rgb"][offset] for offset in depth_offsets]

            # Get features for panoptics and depth (joint backbone)
            if self.depth_algo is not None:
                # For unsupervised training of depth, we always need triplets
                po_depth_feats = [
                    self.backbone_po_depth(in_data["rgb"][offset]) for offset in depth_offsets
                ]
            else:
                # If we do not train depth, only consider the center image
                po_depth_feats = [self.backbone_po_depth(in_data["rgb"][0])]

            # ----------
            # DEPTH TRAINING
            # ----------
            if self.depth_algo is not None:
                # Reconstruct camera model for each sample in the batch
                camera_models = []
                img_height = in_data["rgb"][0].shape[2]
                img_width = in_data["rgb"][0].shape[3]
                for i in range(in_data["rgb"][0].shape[0]):
                    camera_models.append(
                        CameraModel.from_tensor(img_width, img_height, in_data["camera_model"][i]))

                depth_gt = in_data.get("depth")  # Not all datasets provide the GT depth
                depth_recon_loss, depth_smth_loss, flow_smth_loss, flow_sparsity_loss, \
                depth_pred, _, object_motion_maps, depth_stats = self.depth_algo.training(
                    images_window, po_depth_feats, camera_models, self.depth_head,
                    self.backbone_pose_sflow, self.pose_head, self.flow_head, depth_gt)
            else:
                depth_recon_loss, depth_smth_loss, flow_smth_loss, flow_sparsity_loss, depth_pred, \
                object_motion_maps, depth_stats = None, None, None, None, None, None, None

            # ----------
            # SEMANTICS TRAINING
            # ----------
            if self.semantic_algo is not None:
                semantic_gt_eval = in_data.get("semantic_eval")
                semantic_loss, confusion_matrix, semantic_pred = self.semantic_algo.training(
                    po_depth_feats[0], self.semantic_head, in_data["semantic"],
                    in_data["semantic_weights"], sem_ignore_classes, semantic_gt_eval)
            else:
                semantic_loss, confusion_matrix, semantic_pred = None, None, None
            semantic_soft_pred = None

            # ----------
            # INSTANCE TRAINING
            # ----------
            if self.instance_algo is not None:
                # if self.instance_head.use_thing_mask:
                center_loss, offset_loss, center_pred, offset_pred, thing_mask_loss, \
                thing_mask_pred = \
                    self.instance_algo.training(po_depth_feats[0], center=in_data["center"],
                                                offset=in_data["offset"],
                                                center_weights=in_data["center_weights"],
                                                offset_weights=in_data["offset_weights"],
                                                thing_mask=in_data["thing_mask"],
                                                instance_head=self.instance_head)

            else:
                center_loss, offset_loss, center_pred, offset_pred = None, None, None, None
                thing_mask_loss, thing_mask_pred = None, None

        # In evaluation mode, return the predictions and the statistics. But not the loss.
        elif mode == "eval":
            # Get features for panoptics and depth (joint backbone)
            key_rgb = "rgb" if "rgb" in in_data else "rgb_tgt"  # Account for mixup
            po_depth_feats = self.backbone_po_depth(in_data[key_rgb][0])

            if self.depth_algo is not None:
                depth_gt = in_data.get("depth")  # Not all datasets provide the GT depth
                if depth_gt is None:
                    depth_pred = self.depth_algo.inference(po_depth_feats, self.depth_head)
                    depth_stats = None
                else:
                    depth_stats, depth_pred = self.depth_algo.evaluation(po_depth_feats,
                                                                         self.depth_head,
                                                                         depth_gt)
            else:
                depth_stats, depth_pred = None, None
            depth_recon_loss, depth_smth_loss, flow_smth_loss, flow_sparsity_loss, \
            object_motion_maps = None, None, None, None, None

            if self.semantic_algo is not None:
                semantic_gt = in_data.get("semantic")  # Not all samples have GT semantic
                if semantic_gt is None:
                    semantic_pred, semantic_soft_pred = self.semantic_algo.inference(
                        po_depth_feats, self.semantic_head)
                    confusion_matrix = None
                else:
                    semantic_gt = in_data.get("semantic_eval", semantic_gt)
                    confusion_matrix, semantic_pred, semantic_soft_pred = \
                        self.semantic_algo.evaluation(po_depth_feats, self.semantic_head,
                                                      semantic_gt)
            else:
                confusion_matrix, semantic_pred, semantic_soft_pred = None, None, None
            semantic_loss = None

            if self.instance_algo is not None:
                center_pred, offset_pred, thing_mask_pred = \
                    self.instance_algo.inference(po_depth_feats, self.instance_head)
            else:
                center_pred, offset_pred, thing_mask_pred = None, None, None
            center_loss, offset_loss, thing_mask_loss = None, None, None

        # In inference mode, only return the predictions
        elif mode == "infer":
            # Get features for panoptics and depth (joint backbone)
            po_depth_feats = self.backbone_po_depth(in_data["rgb"][0])

            if self.depth_algo is not None:
                depth_pred = self.depth_algo.inference(po_depth_feats, self.depth_head)
            else:
                depth_pred = None
            depth_recon_loss, depth_smth_loss, flow_smth_loss, flow_sparsity_loss, \
            object_motion_maps, depth_stats = None, None, None, None, None, None

            if self.semantic_algo is not None:
                semantic_pred, semantic_soft_pred = self.semantic_algo.inference(po_depth_feats,
                                                                                 self.semantic_head)
            else:
                semantic_pred, semantic_soft_pred = None, None
            semantic_loss, confusion_matrix = None, None

            if self.instance_algo is not None:
                if self.instance_head.use_thing_mask:
                    center_pred, offset_pred, thing_mask_pred = self.instance_algo.inference(
                        po_depth_feats, self.instance_head)
                else:
                    center_pred, offset_pred, thing_mask_pred = self.instance_algo.inference(
                        po_depth_feats,
                        self.instance_head)
                    thing_mask_pred = None
            else:
                center_pred, offset_pred, thing_mask_pred = None, None, None
            center_loss, offset_loss, thing_mask_loss = None, None, None

        else:
            depth_recon_loss, depth_smth_loss, flow_smth_loss, flow_sparsity_loss, depth_pred, \
            object_motion_maps, depth_stats, semantic_loss, confusion_matrix, semantic_pred, \
            semantic_soft_pred, center_loss, offset_loss, center_pred, offset_pred, \
            thing_mask_loss, thing_mask_pred = \
                None, None, None, None, None, None, None, None, None, None, None, None, None, \
                None, None, None, None

        # --------------------
        if do_panoptic_fusion:
            panoptic_pred, instance_pred = self.instance_algo.panoptic_fusion(
                semantic_pred, center_pred, offset_pred)
        else:
            panoptic_pred, instance_pred = None, None

        # --------------------
        # Compute depth metrics for each class
        if do_class_wise_depth_stats and depth_pred is not None and "semantic" in in_data:
            depth_stats_classes = self.depth_algo.evaluator.compute_depth_metrics_per_class(
                in_data["depth"], depth_pred, in_data["semantic"])
        else:
            depth_stats_classes = None

        # --------------------
        # Losses
        losses = OrderedDict()
        losses["depth_recon"] = depth_recon_loss
        losses["depth_smth"] = depth_smth_loss
        losses["flow_smth"] = flow_smth_loss
        losses["flow_sparsity"] = flow_sparsity_loss
        losses["semantic"] = semantic_loss
        losses["center"] = center_loss
        losses["offset"] = offset_loss
        losses["thing_mask"] = thing_mask_loss

        # Predictions
        result = OrderedDict()
        result["depth"] = depth_pred
        result["object_motion_map"] = object_motion_maps
        result["semantic"] = semantic_pred
        result["semantic_soft"] = semantic_soft_pred
        result["center"] = center_pred
        result["offset"] = offset_pred
        result["panoptic"] = panoptic_pred
        result["instance"] = instance_pred
        result["thing_mask"] = thing_mask_pred

        # Statistics
        stats = OrderedDict()
        stats["sem_conf"] = confusion_matrix
        if depth_stats is not None:
            stats.update(depth_stats)
        if depth_stats_classes is not None:
            stats.update(depth_stats_classes)

        return losses, result, stats

    @classmethod
    def ema_model(cls, model):
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()  # This is an in-place operation
        return ema_model

    def update_weights(self, student_model: nn.Module, modules: List[str], alpha_teacher: float,
                       iteration: Optional[int] = None):
        if iteration is not None:
            alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

        # Update ema params for only the specified modules
        for module in modules:
            for ema_named_param, param in zip(self.named_parameters(), student_model.parameters()):
                if ema_named_param[0].startswith(module):
                    ema_named_param[1].data[:] = alpha_teacher * ema_named_param[1][:].data[:] + \
                                                 (1 - alpha_teacher) * param[:].data[:]

    def adapt(self,
              in_data: Dict[str, Dict[str, Tensor]],
              depth_offsets: Optional[List[int]] = None,
              do_panoptic_fusion: bool = False):
        # We assume depth_offsets = [0, -offset, +offset] with offset as a positive integer
        assert depth_offsets[0] == 0
        assert depth_offsets[1] < 0
        assert depth_offsets[1] == -depth_offsets[2]
        if do_panoptic_fusion:
            assert self.semantic_algo is not None
            assert self.instance_algo is not None

        # Reconstruct camera model for each sample in the batch
        camera_models = {}
        for key, value in in_data.items():
            if not isinstance(value, dict) or key.endswith("mixup"):
                continue
            camera_models[key] = []
            img_height = value["rgb"][0].shape[2]
            img_width = value["rgb"][0].shape[3]
            for i in range(value["rgb"][0].shape[0]):
                camera_models[key].append(
                    CameraModel.from_tensor(img_width, img_height, value["camera_model"][i]))

        # ToDo: here again, we assume there is only 1 semantic mixup image (which actually makes
        #  sense) rather then for all offsets in depth_offsets
        # Get images for the window
        images_window = {
            key: [value["rgb"][offset] for offset in depth_offsets if not key.endswith("mixup")] for
            key, value in in_data.items() if isinstance(value, dict)
        }

        # Get features for panoptics and depth (joint backbone)
        # For unsupervised training of depth and semantic, we always need triplets
        po_depth_feats = {
            key: [self.backbone_po_depth(rgb_offset) for rgb_offset in value["rgb"].values()] for
            key, value in in_data.items() if isinstance(value, dict) if not key.endswith("mixup")
        }

        # Apply mixup
        for key in in_data.keys():
            if key.endswith("mixup"):
                in_data[key] = Mixup.do_mixup(key, in_data[key], self.instance_algo, self.cfg_mixup)
                # Update po_depth_feats by mixup features
                po_depth_feats.update({key: [self.backbone_po_depth(in_data[key]["rgb"][0])]})

        # ----------
        # INSTANCE ADAPTATION
        # ----------
        if self.instance_algo is not None:
            center_gt = {key: value.get("center") for key, value in in_data.items() if
                         isinstance(value, dict)}
            offset_gt = {key: value.get("offset") for key, value in in_data.items() if
                         isinstance(value, dict)}
            center_weights = {key: value.get("center_weights") for key, value in in_data.items() if
                              isinstance(value, dict)}
            offset_weights = {key: value.get("offset_weights") for key, value in in_data.items() if
                              isinstance(value, dict)}
            thing_mask = {key: value.get("thing_mask") for key, value in in_data.items() if
                          isinstance(value, dict)}
            center_ema_det = in_data["target"].get("center_ema", None)
            offset_ema_det = in_data["target"].get("offset_ema", None)

            if self.instance_head.use_thing_mask:
                center_losses, offset_losses, center_pred, offset_pred, thing_mask_losses, \
                thing_mask_pred = self.instance_algo.adaptation(po_depth_feats,
                                                                center_gt, offset_gt,
                                                                center_weights, offset_weights,
                                                                thing_mask, self.instance_head,
                                                                center_ema_det, offset_ema_det)
            else:
                center_losses, offset_losses, center_pred, offset_pred = \
                    self.instance_algo.adaptation(po_depth_feats, center_gt, offset_gt,
                                                  center_weights, offset_weights, thing_mask,
                                                  self.instance_head, center_ema_det,
                                                  offset_ema_det)
                thing_mask_losses, thing_mask_pred = {}, None
        else:
            center_losses, offset_losses, center_pred, offset_pred = {}, {}, None, None
            thing_mask_losses, thing_mask_pred = {}, None

        # ----------
        # SEMANTICS ADAPTATION
        # ----------
        if self.semantic_algo is not None:
            semantic_gt = {key: value.get("semantic") for key, value in in_data.items() if
                           isinstance(value, dict)}
            semantic_weights = {key: value.get("semantic_weights") for key, value in in_data.items()
                                if isinstance(value, dict)}

            semantic_losses, confusion_matrix, semantic_pred = self.semantic_algo.adaptation(
                po_depth_feats, self.semantic_head, semantic_gt, semantic_weights)
        else:
            semantic_losses, confusion_matrix, semantic_pred = {}, None, None

        # --------------------
        if do_panoptic_fusion:
            panoptic_pred, instance_pred = self.instance_algo.panoptic_fusion(
                semantic_pred, center_pred, offset_pred)
        else:
            panoptic_pred, instance_pred = None, None
        # --------------------

        # ----------
        # DEPTH ADAPTATION
        # ----------
        if self.depth_algo is not None:
            depth_losses, flow_losses, depth_pred, transformations, object_motion_maps, \
            depth_stats = self.depth_algo.adaptation(images_window, po_depth_feats, camera_models,
                                                     self.depth_head, self.backbone_pose_sflow,
                                                     self.pose_head, self.flow_head)
        else:
            depth_losses, flow_losses, depth_pred, transformations, object_motion_maps, \
            depth_stats = {}, {}, None, None, None, None

        # --------------------
        # Losses
        losses = OrderedDict()
        losses["depth_recon"] = depth_losses.get("recon", None)
        losses["depth_smth"] = depth_losses.get("smth", None)
        losses["flow_smth"] = flow_losses.get("smth", None)
        losses["flow_sparsity"] = flow_losses.get("sparsity", None)
        losses["semantic_source"] = semantic_losses["source"]
        losses["semantic_cut_mixup"] = semantic_losses.get("cut_mixup", None)
        losses["center_source"] = center_losses.get("source", None)
        losses["offset_source"] = offset_losses.get("source", None)

        # Predictions
        result = OrderedDict()
        result["depth"] = depth_pred
        result["object_motion_map"] = object_motion_maps
        result["semantic"] = semantic_pred
        result["center"] = center_pred
        result["offset"] = offset_pred
        result["panoptic"] = panoptic_pred
        result["instance"] = instance_pred
        result["thing_mask"] = thing_mask_pred
        result["image_features"] = po_depth_feats["target"][0][-1].detach()

        # Statistics
        stats = OrderedDict()
        stats["sem_conf"] = confusion_matrix
        if depth_stats is not None:
            stats.update(depth_stats)

        return losses, result, stats, in_data

    def get_state_dict(self) -> Dict[str, Any]:

        def _safe_state_dict(module):
            if module is None:
                return None
            return module.state_dict()

        state_dict = {
            "backbone_po_depth": _safe_state_dict(self.backbone_po_depth),
            "backbone_pose_sflow": _safe_state_dict(self.backbone_pose_sflow),
            "depth_head": _safe_state_dict(self.depth_head),
            "pose_head": _safe_state_dict(self.pose_head),
            "flow_head": _safe_state_dict(self.flow_head),
            "semantic_head": _safe_state_dict(self.semantic_head),
            "instance_head": _safe_state_dict(self.instance_head),
        }

        return state_dict
