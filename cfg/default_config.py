# Reference point for all configurable options
from yacs.config import CfgNode as CN

# /----- Create a cfg node
cfg = CN()

# ********************************************************************
# /------ Training parameters
# ********************************************************************
cfg.train = CN()
cfg.train.nof_epochs = 20
cfg.train.nof_workers_per_gpu = 1
cfg.train.batch_size_per_gpu = 1

# /----- Optimizer parameters
cfg.train.optimizer = CN()
cfg.train.optimizer.type = 'Adam'
cfg.train.optimizer.learning_rate = 0.0001

# /----- Scheduler parameters
cfg.train.scheduler = CN()
cfg.train.scheduler.type = 'StepLR'  # 'StepLR', 'WarmupPolyLR'
# StepLR
cfg.train.scheduler.step_lr = CN()
cfg.train.scheduler.step_lr.step_size = 20
cfg.train.scheduler.step_lr.gamma = 0.1
# WarmupPolyLR
cfg.train.scheduler.warmup = CN()
cfg.train.scheduler.warmup.max_iters = 90000
cfg.train.scheduler.warmup.factor = 0.001
cfg.train.scheduler.warmup.iters = 1000
cfg.train.scheduler.warmup.method = 'linear'
cfg.train.scheduler.warmup.power = 0.9
cfg.train.scheduler.warmup.constant_ending = 0.

# ********************************************************************
# /------ Validation parameters
# ********************************************************************
cfg.val = CN()
cfg.val.batch_size_per_gpu = 1
cfg.val.nof_workers_per_gpu = 1

# ********************************************************************
# /----- Model parameters
# ********************************************************************
cfg.model = CN()

cfg.model.make_depth = True
cfg.model.make_sflow = False
cfg.model.make_semantic = True
cfg.model.make_instance = True

cfg.model.po_depth_net = CN()
cfg.model.po_depth_net.params = CN()
cfg.model.po_depth_net.params.nof_layers = 101
cfg.model.po_depth_net.params.weights_init = 'pretrained'

cfg.model.pose_sflow_net = CN()
cfg.model.pose_sflow_net.input = 'pairs'
cfg.model.pose_sflow_net.params = CN()
cfg.model.pose_sflow_net.params.nof_layers = 18
cfg.model.pose_sflow_net.params.weights_init = 'pretrained'

cfg.model.semantic_head = CN()
cfg.model.semantic_head.use_guda_fusion = True

cfg.model.instance_head = CN()
cfg.model.instance_head.use_thing_mask = False

# ********************************************************************
# /----- Dataset parameters
# ********************************************************************
cfg.dataset = CN()
cfg.dataset.name = ''  # e.g., 'kitti_360'
cfg.dataset.path = ''  # e.g., '/home/shared/ondap/data/kitti_360'
cfg.dataset.feed_img_size = []  # [height, width], e.g., [192, 640]
cfg.dataset.offsets = [1]  # e.g., [1, 2] --> return images [-2, -1, 0, 1, 2]
cfg.dataset.center_heatmap_sigma = 8
cfg.dataset.return_only_rgb = False
cfg.dataset.small_instance_area_full_res = 4096
cfg.dataset.small_instance_weight = 3
cfg.dataset.train_split = 'train'
cfg.dataset.train_sequences = []  # Only supported in 'sequence' split
cfg.dataset.val_split = 'val'
cfg.dataset.val_sequences = []  # Only supported in 'sequence' split
cfg.dataset.remove_classes = []
cfg.dataset.label_mode = 'codeps' # 'cityscapes', 'codeps'

# ********************************************************************
# /----- Preprocessing parameters
# ********************************************************************
cfg.dataset.augmentation = CN()
cfg.dataset.augmentation.active = True  # Whether to apply augmentation
cfg.dataset.augmentation.horizontal_flipping = True  # Randomly applied with prob=.5
cfg.dataset.augmentation.brightness_jitter = 0.2  # Or None
cfg.dataset.augmentation.contrast_jitter = 0.2  # Or None
cfg.dataset.augmentation.saturation_jitter = 0.2  # Or None
cfg.dataset.augmentation.hue_jitter = 0.1  # Or None

cfg.dataset.normalization = CN()
cfg.dataset.normalization.active = True
cfg.dataset.normalization.rgb_mean = (0.485, 0.456, 0.406)
cfg.dataset.normalization.rgb_std = (0.229, 0.224, 0.225)

# ********************************************************************
# /----- Evaluation parameters
# ********************************************************************
cfg.eval = CN()
cfg.eval.depth = CN()
cfg.eval.depth.use_gt_scale = True
cfg.eval.depth.depth_ranges = [0.1, 80]

cfg.eval.semantic = CN()
cfg.eval.semantic.ignore_classes = []

# ********************************************************************
# /----- Losses
# ********************************************************************
cfg.losses = CN()

cfg.losses.weights = CN()
cfg.losses.weights.depth_recon = 1.0
cfg.losses.weights.depth_smth = 0.0001
cfg.losses.weights.flow_smth = 1.0
cfg.losses.weights.flow_sparsity = 1.0

cfg.losses.weights.semantic = 1.0

cfg.losses.weights.center = 1.0
cfg.losses.weights.offset = 1.0
cfg.losses.weights.thing_mask = 1.0

# ********************************************************************
# /----- Semantics
# ********************************************************************
cfg.semantics = CN()
cfg.semantics.class_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0)
cfg.semantics.top_k = 0.2

# ********************************************************************
# /----- Depth
# ********************************************************************
cfg.depth = CN()
cfg.depth.num_recon_scales = 5

# ********************************************************************
# /----- Visualization
# *******************************************************************
cfg.visualization = CN()
cfg.visualization.scale = 1.  # Size of images on wandb

# ********************************************************************
# /----- Logging
# *******************************************************************
cfg.logging = CN()
cfg.logging.log_train_samples = True
# Number of epochs between validations
cfg.logging.val_epoch_interval = 1
# Number of steps before outputting a log entry
cfg.logging.log_step_interval = 10

# ********************************************************************
# /----- Additional parameters from PoBev
# ********************************************************************
cfg.general = CN()
cfg.general.cudnn_benchmark = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()
