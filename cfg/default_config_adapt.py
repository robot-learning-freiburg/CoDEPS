# Reference point for all configurable options
from yacs.config import CfgNode as CN

# /----- Create a cfg node
cfg = CN()

# ********************************************************************
# /------ Adaptation parameters
# ********************************************************************
cfg.adapt = CN()
cfg.adapt.mode = 'online' # 'online' OR 'off' (disables adaptation)

# /----- Adapt these models (if they are activated in cfg.model)
cfg.adapt.model = CN()
cfg.adapt.model.backbone_po_depth = True
cfg.adapt.model.backbone_pose_sflow = True
cfg.adapt.model.depth = True
cfg.adapt.model.pose = True
cfg.adapt.model.sflow = True
cfg.adapt.model.semantic = True
cfg.adapt.model.instance = True

# /----- Use EMA for these models (if they are activated above)
cfg.adapt.ema = CN()
cfg.adapt.ema.alpha = .999
cfg.adapt.ema.depth = False  # Depth head, pose backbone, pose head
cfg.adapt.ema.semantic = False
cfg.adapt.ema.instance = False

cfg.train = CN()
cfg.train.nof_adaptation_steps = 1  # NOTE: only valid in online adaptation
cfg.train.nof_workers_per_gpu = 10

# /----- Optimizer parameters
cfg.train.optimizer = CN()
cfg.train.optimizer.type = 'Adam'
cfg.train.optimizer.learning_rate = 0.0001

# ********************************************************************
# /------ Validation parameters
# ********************************************************************
cfg.val = CN()
cfg.val.batch_size_per_gpu = 2
cfg.val.nof_workers_per_gpu = 10

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
# /----- Target dataset parameters
# ********************************************************************
cfg.dataset = CN()
cfg.dataset.name = ''  # e.g., 'kitti_360'
cfg.dataset.path = ''  # e.g., '/home/shared/ondap/data/kitti_360'
cfg.dataset.sequences = []  # NOTE: only valid for online adaptation mode
cfg.dataset.feed_img_size = []  # [height, width], e.g., [192, 640]
cfg.dataset.offsets = [1]  # e.g., [1, 2] --> return images [-2, -1, 0, 1, 2]
cfg.dataset.batch_size_per_gpu = 2  # NOTE: automatically set to 1 for online/off adaptation mode
cfg.dataset.center_heatmap_sigma = 8
cfg.dataset.return_only_rgb = False
cfg.dataset.small_instance_area_full_res = 4096
cfg.dataset.small_instance_weight = 3
cfg.dataset.remove_classes = []  # ID of classes to be removed
cfg.dataset.label_mode = 'codeps' # 'cityscapes', 'codeps'

# ********************************************************************
# /----- Preprocessing parameters
# ********************************************************************
cfg.dataset.augmentation = CN()
cfg.dataset.augmentation.active = True  # Whether to apply additional augmentation
cfg.dataset.augmentation.horizontal_flipping = False  # Randomly applied with prob=.5
cfg.dataset.augmentation.brightness_jitter = 0.2  # Or None
cfg.dataset.augmentation.contrast_jitter = 0.2  # Or None
cfg.dataset.augmentation.saturation_jitter = 0.2  # Or None
cfg.dataset.augmentation.hue_jitter = 0.1  # Or None

cfg.dataset.normalization = CN()
cfg.dataset.normalization.active = True
cfg.dataset.normalization.rgb_mean = (0.485, 0.456, 0.406)
cfg.dataset.normalization.rgb_std = (0.229, 0.224, 0.225)

# ********************************************************************
# /----- Source dataset and replay buffer parameters
# ********************************************************************
cfg.adapt.source_dataset = CN()
cfg.adapt.source_dataset.name = ''
cfg.adapt.source_dataset.path = ''
cfg.adapt.source_dataset.feed_img_size = []  # [height, width], e.g., [256, 512]
cfg.adapt.source_dataset.offsets = [1]  # e.g., [1, 2] --> return images [-2, -1, 0, 1, 2]

# /----- Replay buffer parameters
cfg.adapt.replay_sampler = CN()
cfg.adapt.replay_sampler.nof_source_samples = 1 # Number of items sampled from replay buffer
cfg.adapt.replay_sampler.nof_target_samples = 0
cfg.adapt.replay_sampler.seed = 42
cfg.adapt.replay_buffer = CN()
cfg.adapt.replay_buffer.source_size = None # Sampled randomly to add to replay buffer, None = inf
cfg.adapt.replay_buffer.target_size = None # None = inf
cfg.adapt.replay_buffer.maximize_diversity = False
cfg.adapt.replay_buffer.similarity_threshold = 0.95
cfg.adapt.replay_buffer.seed = 42

# Ratio of samples for adaptation, the others are used for evaluation
cfg.adapt.target_dataset_adapt_ratio = 0.7

# Mix-up parameters
cfg.adapt.mixup = CN()
cfg.adapt.mixup.general = CN()
cfg.adapt.mixup.general.active = True
# "cut_mixup", """class_mixup" ""'conf_instance_mixup',
cfg.adapt.mixup.general.mixup_strategies = ['cut_mixup', 'conf_instance_mixup']
cfg.adapt.mixup.general.nof_samples = 2
cfg.adapt.mixup.general.geom_augment = True

cfg.adapt.mixup.cut_mix = CN()
cfg.adapt.mixup.cut_mix.nof_hor_splits = 4
cfg.adapt.mixup.cut_mix.nof_vert_splits = 4
cfg.adapt.mixup.cut_mix.nof_segments = 1

cfg.adapt.mixup.conf_instance_mix = CN()
cfg.adapt.mixup.conf_instance_mix.conf_thresh = 0.0
cfg.adapt.mixup.conf_instance_mix.min_inst_size = 0

cfg.adapt.mixup.class_mix = CN()
cfg.adapt.mixup.class_mix.conf_thresh = 0.0 # ToDo: This is not used yet

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
# Losses can be disabled by setting them to None
cfg.losses = CN()
cfg.losses.weights = CN()

cfg.losses.weights.depth_recon = 10.0
cfg.losses.weights.depth_smth = 0.001

cfg.losses.weights.flow_smth = 10.0
cfg.losses.weights.flow_sparsity = 10.0

cfg.losses.weights.semantic_source = 1.0
cfg.losses.weights.semantic_cut_mixup = 1.0

cfg.losses.weights.center_source = 20.0
cfg.losses.weights.offset_source = 0.1

# ********************************************************************
# /----- Depth
# ********************************************************************
cfg.depth = CN()
cfg.depth.num_recon_scales = 5

# ********************************************************************
# /----- Semantics
# ********************************************************************
cfg.semantics = CN()
cfg.semantics.class_weights = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
cfg.semantics.top_k = 0.2

# ********************************************************************
# /----- Visualization
# *******************************************************************
cfg.visualization = CN()
cfg.visualization.scale = .5  # Size of images on wandb

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
