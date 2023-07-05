from typing import List, Optional

from algos import (
    BinaryMaskLoss,
    CenterLoss,
    DepthAlgo,
    EdgeAwareSmoothnessLoss,
    FlowSmoothnessLoss,
    FlowSparsityLoss,
    InstanceSegAlgo,
    OffsetLoss,
    ReconstructionLoss,
    SemanticConsistencyLoss,
    SemanticLoss,
    SemanticSegAlgo,
    SSIMLoss,
)
from codeps.online_adap import CodepsNet
from eval import DepthEvaluator, PanopticEvaluator, SemanticEvaluator
from models import (
    DepthHead,
    FlowHead,
    InstanceHead,
    PoseHead,
    ResnetEncoder,
    SemanticHead,
)


def gen_models(cfg, device, stuff_classes: List[int], thing_classes: List[int],
               ignore_classes: List[int], label_mode: Optional[str]=None,
               adaptation_mode: bool = False) -> CodepsNet:
    """Create the backbones, heads, losses, evaluators, and algorithms
    """

    ref_img_height = {"target": cfg.dataset.feed_img_size[0], "source": None}
    ref_img_width = {"target": cfg.dataset.feed_img_size[1], "source": None}
    if adaptation_mode:
        ref_img_height["source"] = cfg.adapt.source_dataset.feed_img_size[0]
        ref_img_width["source"] = cfg.adapt.source_dataset.feed_img_size[1]

    # ----------
    # MULTI-TASK BACKBONE
    # ----------
    backbone_po_depth = ResnetEncoder(cfg.model.po_depth_net.params.nof_layers,
                                      cfg.model.po_depth_net.params.weights_init == "pretrained")

    # ----------
    # DEPTH + POSE + SFLOW SETUP
    # ----------
    if cfg.model.make_depth:
        num_channels_input = 4 if cfg.model.make_sflow else 3  # RGB-D vs. RGB
        backbone_pose_sflow = ResnetEncoder(
            cfg.model.pose_sflow_net.params.nof_layers,
            cfg.model.pose_sflow_net.params.weights_init == "pretrained",
            num_input_images=2,
            num_channels_input=num_channels_input)
        depth_head = DepthHead(backbone_po_depth.num_ch_enc, use_skips=True)
        pose_head = PoseHead(backbone_pose_sflow.num_ch_enc,
                             num_input_features=1,
                             num_frames_to_predict_for=2)

        ssim_loss = SSIMLoss()
        depth_rec_loss = {
            "target": ReconstructionLoss(ref_img_width["target"], ref_img_height["target"],
                                         ssim_loss, cfg.depth.num_recon_scales, device),
            "source": None
        }
        if adaptation_mode:
            depth_rec_loss["source"] = ReconstructionLoss(ref_img_width["source"],
                                                          ref_img_height["source"], ssim_loss,
                                                          cfg.depth.num_recon_scales, device)
        depth_smth_loss = EdgeAwareSmoothnessLoss()

        if cfg.model.make_sflow:
            flow_head = FlowHead(backbone_pose_sflow.num_ch_enc)
            flow_smth_loss = FlowSmoothnessLoss()
            flow_sparsity_loss = FlowSparsityLoss()
        else:
            flow_head, flow_smth_loss, flow_sparsity_loss = None, None, None

        depth_eval = DepthEvaluator(cfg.eval.depth.use_gt_scale, cfg.eval.depth.depth_ranges)
        depth_algo = DepthAlgo(depth_rec_loss["target"], depth_smth_loss, depth_eval,
                               flow_smth_loss, flow_sparsity_loss, depth_rec_loss["source"],
                               label_mode)
    else:
        backbone_pose_sflow, depth_head, pose_head, flow_head = None, None, None, None
        depth_algo = None

    # ----------
    # SEMANTICS SETUP
    # ----------
    if cfg.model.make_semantic:
        num_classes = len(stuff_classes) + len(thing_classes)
        semantic_head = SemanticHead(backbone_po_depth.num_ch_enc,
                                     num_classes,
                                     use_skips=True,
                                     use_guda_fusion=cfg.model.semantic_head.use_guda_fusion)
        # Remove weights that belong to cfg.dataset.remove_classes
        class_weights = [wt for idx, wt in enumerate(cfg.semantics.class_weights) if idx not in
                         cfg.dataset.remove_classes]
        sem_loss = SemanticLoss(device=device, class_weights=class_weights,
                                top_k_percent_pixels=cfg.semantics.top_k,
                                ignore_labels=ignore_classes)
        if adaptation_mode:
            sem_consistency_loss = SemanticConsistencyLoss(device, ref_img_width["target"],
                                                           ref_img_height["target"])
        else:
            sem_consistency_loss = None
        sem_eval = SemanticEvaluator(num_classes=num_classes, ignore_classes=ignore_classes)
        sem_algo = SemanticSegAlgo(sem_loss, sem_eval, sem_consistency_loss)
    else:
        semantic_head, sem_algo = None, None

    # ----------
    # INSTANCE SETUP
    # ----------
    if cfg.model.make_instance:
        instance_head = InstanceHead(backbone_po_depth.num_ch_enc,
                                     use_thing_mask=cfg.model.instance_head.use_thing_mask)
        instance_center_loss = CenterLoss()
        instance_offset_loss = OffsetLoss()
        binary_mask_loss = BinaryMaskLoss()
        panoptic_eval = PanopticEvaluator(stuff_list=stuff_classes,
                                          thing_list=thing_classes,
                                          label_divisor=1000, void_label=-1)
        instance_algo = InstanceSegAlgo(instance_center_loss, instance_offset_loss, panoptic_eval,
                                        binary_mask_loss)
    else:
        instance_head, instance_algo = None, None

    # ----------
    # OVERALL NETWORK
    # ----------
    codeps_net = CodepsNet(cfg_mixup=cfg.adapt.mixup,
                          backbone_po_depth=backbone_po_depth,
                          backbone_pose_sflow=backbone_pose_sflow,
                          depth_head=depth_head,
                          pose_head=pose_head,
                          flow_head=flow_head,
                          semantic_head=semantic_head,
                          instance_head=instance_head,
                          depth_algo=depth_algo,
                          semantic_algo=sem_algo,
                          instance_algo=instance_algo)

    return codeps_net
