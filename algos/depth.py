from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from datasets import get_labels
from eval import DepthEvaluator
from misc import CameraModel, ImageWarper, Laplace
from models import DepthHead, FlowHead, PoseHead, ResnetEncoder

# --------------------- Flow losses ---------------------- #


class FlowSmoothnessLoss:

    def __init__(self, wrap_around: bool = True):
        self.wrap_around = wrap_around

    def _compute_loss(self, flow_map: Tensor) -> Tensor:
        grad_x = flow_map - torch.roll(flow_map, shifts=1, dims=3)
        grad_y = flow_map - torch.roll(flow_map, shifts=1, dims=2)
        if not self.wrap_around:
            grad_x = grad_x[:, :, 1:, 1:]
            grad_y = grad_y[:, :, 1:, 1:]
        loss = torch.mean(torch.sqrt(torch.square(grad_x) + torch.square(grad_y) + 1e-7))
        return loss

    def __call__(self, flow_maps: Tuple[Tensor, ...]) -> Tensor:
        loss = self._compute_loss(flow_maps[0])
        for flow_map in flow_maps[1:]:
            loss += self._compute_loss(flow_map)
        loss /= len(flow_maps)
        return loss


class FlowSparsityLoss:

    @staticmethod
    def _compute_loss(flow_map: Tensor) -> Tensor:
        abs_map = torch.abs(flow_map)
        spatial_mean_motion = torch.mean(abs_map, dim=(2, 3), keepdim=True).detach()
        loss = torch.mean(2 * spatial_mean_motion * torch.sqrt(abs_map /
                                                               (spatial_mean_motion + 1e-7) + 1))
        return loss

    def __call__(self, flow_maps: Tuple[Tensor, ...]) -> Tensor:
        loss = self._compute_loss(flow_maps[0])
        for flow_map in flow_maps[1:]:
            loss += self._compute_loss(flow_map)
        loss /= len(flow_maps)
        return loss


# --------------------- Depth losses --------------------- #


class EdgeAwareSmoothnessLoss:
    """Edge-aware smoothness loss
    """

    def __init__(self):
        pass

    @staticmethod
    def _compute_loss(disp: Tensor, img: Tensor) -> Tensor:
        """Compute the edge-aware smoothness loss for a normalized disparity image.
        Parameters
        ----------
        disp : torch.Tensor
            The normalized disparity image
        img : torch.Tensor
            The corresponding RGB image used to consider edge-aware smoothness
        Returns
        -------
        loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        loss_x = grad_disp_x * torch.exp(-grad_img_x)
        loss_y = grad_disp_y * torch.exp(-grad_img_y)
        loss = loss_x.mean() + loss_y.mean()

        return loss

    def __call__(self, target_image: Tensor, disparity_map: Tensor) -> Tensor:
        """Compute the edge-aware smoothness loss for a disparity image.
        Parameters
        ----------
        disp : torch.Tensor
            The disparity image, i.e., the inverse depth
        img : torch.Tensor
            The corresponding RGB image used to consider edge-aware smoothness
        Returns
        -------
        loss : torch.Tensor
            A scalar tensor with the computed loss multiplied by the smoothness factor
        """
        mean_disparity = disparity_map.mean(2, True).mean(3, True)
        norm_disparity = disparity_map / (mean_disparity + 1e-7)
        loss = self._compute_loss(norm_disparity, target_image).sum()
        return loss


class SSIMLoss:
    """Structural Similarity Index (SSIM)
    Parameters
    ----------
    window_size : int
        The size of the moving window used for average pooling
    """

    def __init__(self, window_size: int = 3):
        padding = window_size // 2

        self.mu_pool = nn.AvgPool2d(window_size, padding)
        self.sig_pool = nn.AvgPool2d(window_size, padding)
        self.reflection = nn.ReflectionPad2d(1)

        self.c1 = .01 ** 2
        self.c2 = .03 ** 2

    def __call__(self, src_img: Tensor, target_img: Tensor) -> Tensor:
        """Compute the SSIM loss between a source and a target image
        Parameters
        ----------
        src_img : torch.Tensor
            The source image
        target_img : torch.Tensor
            The corresponding target image matching the dimensions of the source
        Returns
        -------
        loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        x = self.reflection(src_img)
        y = self.reflection(target_img)

        mu_x = self.mu_pool(x)
        mu_y = self.mu_pool(y)

        sigma_x = self.sig_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_pool(x * y) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2)
        loss = torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)

        return loss


class Interpolate(nn.Module):
    """Helper module wrapping the interpolate function of PyTorch functional API
    """

    def __init__(self, width: int, height: int, mode: str):
        super().__init__()
        self.interp = F.interpolate
        self.size = (height, width)
        self.mode = mode

    def forward(self, x):
        if self.mode == "nearest":
            x = self.interp(x, self.size, mode=self.mode)
        else:
            x = self.interp(x, self.size, mode=self.mode, align_corners=False)
        return x


class ReconstructionLoss:
    """Reconstruction loss
    Parameters
    ----------
    ref_img_width: int
        The original image width as it is fed into the network (e.g. 640 for KITTI)
    ref_img_height: int
        The original image height as it is fed into the network (e.g. 192 for KITTI)
    ssim : SSIMLoss
        An instance of the SSIM loss class
    device : torch.device
        Device (GPU) used during training
    num_scales : int
        Number of scales
    alpha : float
        Weight of the SSIM loss, the L1 loss is multiplied by (1 - alpha)
    """

    def __init__(self,
                 ref_img_width,
                 ref_img_height,
                 ssim: SSIMLoss,
                 num_scales: int,
                 device: torch.device,
                 alpha: float = .85):
        self.ssim = ssim
        self.device = device
        self.num_scales = num_scales
        self.alpha = alpha

        self.image_warpers = {}
        self.image_scalers = {}
        self.image_scalers_nearest = {}
        self.scaled_width = {}
        self.scaled_height = {}
        for i in range(self.num_scales):
            scale = 2 ** i
            self.scaled_width[i] = ref_img_width // scale
            self.scaled_height[i] = ref_img_height // scale
            self.image_warpers[i] = ImageWarper(self.scaled_width[i], self.scaled_height[i], device)
            self.image_scalers[i] = Interpolate(self.scaled_width[i], self.scaled_height[i],
                                                "bilinear")
            self.image_scalers_nearest[i] = Interpolate(self.scaled_width[i], self.scaled_height[i],
                                                        "nearest")

    def _compute_loss(self, pred_img: Tensor, target_img: Tensor) -> Tensor:
        """Compute the reconstruction loss between a warped image and the corresponding target.
        Parameters
        ----------
        pred_img : torch.Tensor
            The predicted image, e.g., obtained by image warping
        target_img : torch.Tensor
            The corresponding true image from the same pose.
        Returns
        -------
        loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        l1_loss = torch.abs(pred_img - target_img).mean(1, True)
        ssim_loss = self.ssim(pred_img, target_img).mean(1, True)
        loss = self.alpha * ssim_loss + (1 - self.alpha) * l1_loss
        return loss

    def __call__(
            self,
            camera_models: List[CameraModel],
            images: Tuple[Tensor, Tensor, Tensor],
            depth_map: Tensor,
            poses: Tuple[Tensor, Tensor],
            object_motion_maps: Optional[Tuple[Tensor, Tensor]] = None,
            semantic_mask: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Tensor:
        """Compute the reconstruction loss based on the photometric error
        Unlike Monodepth2, we do not compute the loss on the predicted scales by upscaling but we
        downscale the final depth to different scales.
        Parameters
        ----------
        camera_models : List of CameraModel with length batch_size_per_gpu
            Camera model containing the image intrinsics
        images : Tuple of 3 torch.Tensor
            Triplet of RGB images ordered as [t, t-1, t+1]
        depth_map : torch.Tensor
            Predicted depth map of the image at time t
        poses : Tuple of 2 torch.Tensor
            Predicted camera motion [t -> t-1] and [t -> t+1]
        object_motion_maps : Tuple of 2 torch.Tensor
            Predicted object motion maps of [t -> t-1] and [t -> t+1]
        Returns
        -------
        loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        assert len(camera_models) == images[0].shape[0], "Batch size of camera model does not match"

        loss = torch.zeros(1, device=depth_map.device)

        for s in range(self.num_scales):
            scaled_camera_models = [
                model.get_scaled_model_image_size(self.scaled_width[s], self.scaled_height[s])
                for model in camera_models
            ]

            # Downscale the input RGB image and the predicted depth.
            # Later, we also downscale the neighboring frames and the predicted object motion map.
            scaled_target_img = self.image_scalers[s](images[0])
            scaled_depth_map = self.image_scalers[s](depth_map)

            reconstruction_losses = []
            if semantic_mask is not None:
                scaled_target_sem = self.image_scalers_nearest[s](
                    semantic_mask[0].unsqueeze(1).float())
                for i, frame in enumerate(semantic_mask[1:]):
                    scaled_frame = self.image_scalers_nearest[s](frame.unsqueeze(1).float())
                    pred_sem = self.image_warpers[s](scaled_camera_models, scaled_frame,
                                                     scaled_depth_map, poses[i],
                                                     interp_mode="nearest")
                    reconstruction_losses.append(self._compute_loss(pred_sem, scaled_target_sem))
            else:
                for i, frame in enumerate(images[1:]):
                    scaled_frame = self.image_scalers[s](frame)
                    if object_motion_maps is None:
                        pred_img = self.image_warpers[s](scaled_camera_models, scaled_frame,
                                                         scaled_depth_map, poses[i])
                    else:
                        scaled_object_motion_map = self.image_scalers[s](object_motion_maps[i])
                        pred_img = self.image_warpers[s](scaled_camera_models, scaled_frame,
                                                         scaled_depth_map, poses[i],
                                                         object_motion_map=scaled_object_motion_map)
                    reconstruction_losses.append(self._compute_loss(pred_img, scaled_target_img))
            reconstruction_losses = torch.cat(reconstruction_losses, 1)

            if semantic_mask is not None:
                loss_per_pixel = reconstruction_losses
            else:
                # Auto-masking
                identity_losses = []
                for frame in images[1:]:
                    scaled_frame = self.image_scalers[s](frame)
                    identity_losses.append(self._compute_loss(scaled_frame, scaled_target_img))
                identity_losses = torch.cat(identity_losses, 1)
                # Add random numbers to break ties
                identity_losses += torch.randn(identity_losses.shape, device=depth_map.device) \
                                   * .00001

                combined_losses = torch.cat((reconstruction_losses, identity_losses), dim=1)
                # "minimum among computed losses allows for robust reprojection"
                # https://openaccess.thecvf.com/content_CVPR_2020/papers/Poggi_On_the_Uncertainty_of_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.pdf
                loss_per_pixel, _ = torch.min(combined_losses, dim=1)

            loss += loss_per_pixel.mean() / (2 ** s)
        return loss[0] / self.num_scales


# -------------------------------------------------------- #


class DepthAlgo:
    """Depth algorithm
    Parameters
    ----------
    reconstruction_loss: ReconstructionLoss
        Reconstruction loss by comparing photometric values between the target and warped images
    smoothness_loss: EdgeAwareSmoothnessLoss
        Smoothness loss to regularize the training
    evaluator: DepthEvaluator
        Evaluator for computing depth metrics if GT data is available
    flow_smoothness_loss : Optional[FlowSmoothnessLoss] = None
        Smoothness loss to cover the prior knowledge that the scene flow is mostly constant
    flow_sparsity_loss : Optional[FlowSparsityLoss] = None
        Sparsity loss to cover the prior knowledge that the scene flow is mostly zero
    reconstruction_loss_adapt_source : Optional[ReconstructionLoss] = None
        During adaptation, this loss is used for the source, while the other is for the target.
    """

    def __init__(
            self,
            reconstruction_loss: ReconstructionLoss,
            smoothness_loss: EdgeAwareSmoothnessLoss,
            evaluator: DepthEvaluator,
            flow_smoothness_loss: Optional[FlowSmoothnessLoss] = None,
            flow_sparsity_loss: Optional[FlowSparsityLoss] = None,
            reconstruction_loss_adapt_source: Optional[ReconstructionLoss] = None,
            label_mode: Optional[str] = None,
    ):
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_loss_adapt_source = reconstruction_loss_adapt_source
        self.smoothness_loss = smoothness_loss
        self.evaluator = evaluator
        self.flow_smoothness_loss = flow_smoothness_loss
        self.flow_sparsity_loss = flow_sparsity_loss
        if label_mode is not None:
            self.labels = {label.trainId: label.name for label in get_labels([], label_mode)}
            self.labels[255] = "void"
        else:
            self.labels = None
        self.adaptation_cache = {"target_dist": [], "is_car_moving": False}

    @staticmethod
    def _forward(
            images: Tuple[Tensor, Tensor, Tensor],
            depth_feats_window: Tuple[Tensor, Tensor, Tensor],
            depth_head: DepthHead,
            body_pose_sflow: ResnetEncoder,
            pose_head: PoseHead,
            flow_head: Optional[FlowHead] = None,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], Optional[List[Tensor]]]:
        # Predict the depth maps for all images
        depth_maps, disparity_maps = [], []
        if flow_head is not None:
            for features in depth_feats_window:
                depth, disp = depth_head(features, return_disparity=True)
                depth_maps.append(depth)
                disparity_maps.append(disp)
        else:
            features = depth_feats_window[0]
            depth, disp = depth_head(features, return_disparity=True)
            depth_maps.append(depth)
            disparity_maps.append(disp)

        # Concatenate the predicted depth with the RGB images to generate RGB-D
        # We detach the depth head to avoid any gradient flow from the motion module to the depth
        # estimation module
        if flow_head is not None:
            input_motion_net = [
                torch.cat([img, depth.detach()], dim=1) for img, depth in zip(images, depth_maps)
            ]
        else:
            input_motion_net = list(images)

        # Predict camera motion (and scene flow) for [t -> t-1] and [t -> t+1]
        object_motion_maps, transformations = [], []

        # [t -> t-1]
        # Pass the frames in temporal order but invert the resulting pose
        motion_feats = body_pose_sflow(torch.cat([input_motion_net[1], input_motion_net[0]], dim=1))
        transformations.append(pose_head(motion_feats, invert_pose=True))
        if flow_head is not None:
            object_motion_maps.append(-flow_head(motion_feats))

        # [t -> t+1]
        motion_feats = body_pose_sflow(torch.cat([input_motion_net[0], input_motion_net[2]], dim=1))
        transformations.append(pose_head(motion_feats))
        if flow_head is not None:
            object_motion_maps.append(flow_head(motion_feats))
        else:
            object_motion_maps = None

        return depth_maps, disparity_maps, transformations, object_motion_maps

    def training(
            self,
            images: Tuple[Tensor, Tensor, Tensor],
            depth_feats_window: Tuple[Tensor, Tensor, Tensor],
            camera_models: List[CameraModel],
            depth_head: DepthHead,
            body_pose_sflow: ResnetEncoder,
            pose_head: PoseHead,
            flow_head: Optional[FlowHead] = None,
            depth_gt: Optional[Tensor] = None,
            use_reconstruction_loss_adapt_source: bool = False,
            return_disparity: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, List[Tensor],
               Optional[List[Tensor]], Optional[Dict[str, Tensor]]]:
        """
        Parameters
        ----------
        images : torch.Tensor
            RGB images at times [t, t-1, t+1]
        depth_feats_window : torch.Tensor
            Features from the multi-task encoder used as input to depth, segmentation, etc.
        camera_models : List of CameraModel with length batch_size_per_gpu
            Camera model containing the image intrinsics
        depth_head: DepthHead
            Decoder to predict the depth map based on the input depth features
        body_pose_sflow: ResnetEncoder
            Resnet backbone to predict the features for pose/scene flow prediction
        pose_head: PoseHead
            Decoder to predict the poses between adjacent frames
        flow_head: FlowHead
            Decoder to predict the object motion maps (scene flow) between adjacent frames
        depth_gt: torch.Tensor
            Depth ground truth for computation of depth stats
        use_reconstruction_loss_adapt_source : bool
            Whether to use the reconstruction_loss_adapt_source instead of the normal rec_loss.
        """
        # State all assumptions here
        # - The first image is the reference / target image
        # - Images are from times [t, t-1, t+1]

        depth_maps, disparity_maps, transformations, object_motion_maps = self._forward(
            images, depth_feats_window, depth_head, body_pose_sflow, pose_head, flow_head)

        if depth_gt is not None:
            depth_stats = self.evaluator.compute_depth_metrics(depth_gt, depth_maps[0])
        else:
            depth_stats = None

        # --------------------
        if use_reconstruction_loss_adapt_source:
            depth_recon_loss = self.reconstruction_loss_adapt_source(camera_models, images,
                                                                     depth_maps[0], transformations,
                                                                     object_motion_maps)
        else:
            depth_recon_loss = self.reconstruction_loss(camera_models, images, depth_maps[0],
                                                        transformations, object_motion_maps)
        depth_smth_loss = self.smoothness_loss(images[0], disparity_maps[0])

        if flow_head is not None:
            flow_smth_loss = self.flow_smoothness_loss(object_motion_maps)
            flow_sparsity_loss = self.flow_sparsity_loss(object_motion_maps)
        else:
            flow_smth_loss = None
            flow_sparsity_loss = None

        if return_disparity:
            return depth_recon_loss, depth_smth_loss, flow_smth_loss, flow_sparsity_loss, \
                   depth_maps[0], disparity_maps[0], transformations, object_motion_maps, \
                   depth_stats
        return depth_recon_loss, depth_smth_loss, flow_smth_loss, flow_sparsity_loss, \
               depth_maps[0], transformations, object_motion_maps, depth_stats

    def inference(self, feats: Tensor, depth_head: DepthHead) -> Tensor:
        depth = depth_head(feats)
        return depth

    def evaluation(self, feats: Tensor, depth_head: DepthHead, depth_gt: Tensor) \
            -> Tuple[Dict[str, Tensor], Tensor]:
        depth = self.inference(feats, depth_head)
        depth_stats = self.evaluator.compute_depth_metrics(depth_gt, depth)
        return depth_stats, depth

    def adaptation(
            self,
            images: Dict[str, Tuple[Tensor, Tensor, Tensor]],
            depth_feats_window: Dict[str, Tuple[Tensor, Tensor, Tensor]],
            camera_models: Dict[str, List[CameraModel]],
            depth_head: DepthHead,
            body_pose_sflow: ResnetEncoder,
            pose_head: PoseHead,
            flow_head: Optional[FlowHead] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor, List[Tensor],
               Optional[List[Tensor]], Optional[Dict[str, Tensor]]]:
        # ---------------------------------------
        # Normal unsupervised loss
        KEYS = [key for key in ["source", "target", "target_replay"] if key in images.keys()]

        # The target and source datasets might have different resolutions. Hence, process them
        #  in two forward passes and then average the losses accordingly.
        depth_recon_loss, depth_smth_loss, flow_smth_loss, flow_sparsity_loss = {}, {}, {}, {}
        depth_pred, disparity_pred, transformations, object_motion_maps = {}, {}, {}, {}
        num_images = {key: 0 for key in KEYS}

        for key in KEYS:
            use_reconstruction_loss_adapt_source = key == "source"
            depth_recon_loss[key], depth_smth_loss[key], flow_smth_loss[key], \
            flow_sparsity_loss[key], depth_pred[key], disparity_pred[key], transformations[key], \
            object_motion_maps[key], _ \
                = self.training(images[key], depth_feats_window[key], camera_models[key],
                                depth_head, body_pose_sflow, pose_head, flow_head, None,
                                use_reconstruction_loss_adapt_source, return_disparity=True)

            if key == "target":
                dist = (torch.linalg.norm(transformations[key][0][0, :3, 3]) + torch.linalg.norm(
                    transformations[key][1][0, :3, 3])) / 2
                skip_image = False
                LENGTH = 300
                # Use a rolling buffer to compute the average distance
                if len(self.adaptation_cache["target_dist"]) > LENGTH:
                    self.adaptation_cache["target_dist"].pop(0)
                # Skip the image if the distance is less than 10% of the average distance
                if len(self.adaptation_cache["target_dist"]) == LENGTH and \
                        sum(self.adaptation_cache["target_dist"]) / LENGTH > 10 * dist:
                    skip_image = True
                if skip_image:
                    depth_recon_loss.pop(key)
                    depth_smth_loss.pop(key)
                    flow_smth_loss.pop(key)
                    flow_sparsity_loss.pop(key)
                    print("Skipping image due to non-moving car.")
                else:
                    self.adaptation_cache["target_dist"].append(dist)
                    num_images[key] = images[key][0].shape[0]
                self.adaptation_cache["is_car_moving"] = skip_image
            else:
                num_images[key] = images[key][0].shape[0]

        if sum(num_images.values()) > 0:
            depth_recon_loss = torch.stack(
                [loss * num_images[key] for key, loss in depth_recon_loss.items() if
                 num_images[key] > 0]).sum() / sum(num_images.values())
            depth_smth_loss = torch.stack(
                [loss * num_images[key] for key, loss in depth_smth_loss.items() if
                 num_images[key] > 0]).sum() / sum(num_images.values())
            if flow_head is None:
                flow_smth_loss, flow_sparsity_loss = None, None
            else:
                flow_smth_loss = torch.stack(
                    [loss * num_images[key] for key, loss in flow_smth_loss.items() if
                     num_images[key] > 0]).sum() / sum(num_images.values())
                flow_sparsity_loss = torch.stack(
                    [loss * num_images[key] for key, loss in flow_sparsity_loss.items() if
                     num_images[key] > 0]).sum() / sum(num_images.values())
        else:
            depth_recon_loss = None
            depth_smth_loss = None
            flow_smth_loss = None
            flow_sparsity_loss = None

        # ---------------------------------------
        # Collect return values
        depth_losses = {
            "recon": depth_recon_loss,
            "smth": depth_smth_loss,
        }
        flow_losses = {
            "smth": flow_smth_loss,
            "sparsity": flow_sparsity_loss
        }
        return depth_losses, flow_losses, depth_pred["target"], transformations["target"], \
               object_motion_maps["target"], None
