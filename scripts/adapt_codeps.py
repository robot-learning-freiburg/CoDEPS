# pylint: disable=wrong-import-position

import argparse
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).absolute().parent.parent))

from codeps.model_setup import gen_models
from codeps.online_adap import CodepsNet
from eval.meters import AverageMeter, ConfusionMatrixMeter, ConstantMeter
from io_utils import io_utils, logging
from io_utils.visualizations import gen_visualizations, plot_confusion_matrix
from misc import adapt_utils, train_utils
from scripts.train_codeps import validate

parser = argparse.ArgumentParser(description="Online continual learning on a given dataset")
parser.add_argument("--run_name", required=True, type=str,
                    help="Name of the run")
parser.add_argument("--project_root_dir", required=True, type=str,
                    help="The root directory of the project")
parser.add_argument("--checkpoint", metavar="FILE", type=str,
                    help="Load the pre-trained model weights from the given file")
parser.add_argument("--filename_defaults_config", required=True, type=str,
                    help="Path to defaults configuration file")
parser.add_argument("--filename_config", required=True, type=str,
                    help="Path to configuration file")
parser.add_argument("--comment", type=str,
                    help="Comment to add to WandB")
parser.add_argument("--seed", type=int, default=20,
                    help="Seed to initialize 'torch', 'random', and 'numpy'")
parser.add_argument("--debug", type=bool, default=False,
                    help="Should the program run in 'debug' mode?")


def adapt(model, optimizer, dataloader, frozen_modules, config, device, wandb_summary, debug):
    disable_adaptation = config.adapt.mode == "off"
    adapation_steps = 0 if disable_adaptation else config.train.nof_adaptation_steps
    loss_weights = config.losses.weights
    do_panoptic_fusion = config.model.make_semantic and config.model.make_instance
    do_class_wise_depth_stats = config.model.make_depth and config.model.make_semantic
    log_step_interval = config.logging.log_step_interval
    if config.dataset.normalization.active:
        rgb_mean = config.dataset.normalization.rgb_mean
        rgb_std = config.dataset.normalization.rgb_std
    else:
        rgb_mean, rgb_std = (0., 0., 0.), (0., 0., 0.)
    img_scale = config.visualization.scale
    rgb_frame_offsets = [0] + [-x for x in sorted(config.dataset.offsets, reverse=True)] + sorted(
        config.dataset.offsets)  # e.g., [0, -1, 1]
    ema_alpha = config.adapt.ema.alpha
    ema_modules, ema_modules_inverted = [], []
    if not disable_adaptation:
        ema_modules += ["depth_head", "backbone_pose_sflow",
                        "pose_head"] if config.adapt.ema.depth else []
        ema_modules += ["semantic_head"] if config.adapt.ema.semantic and \
                                            config.adapt.model.semantic else []
        ema_modules += ["instance_head"] if config.adapt.ema.instance and \
                                            config.adapt.model.instance else []
        ema_modules_inverted = ["backbone_po_depth", "backbone_pose_sflow", "depth_head",
                                "pose_head", "semantic_head", "instance_head"]
        for module in ema_modules + frozen_modules:
            if module in ema_modules_inverted:
                ema_modules_inverted.remove(module)
        # ToDo implement the consistency loss for depth
        if config.adapt.ema.depth:
            raise NotImplementedError("Consistency loss for depth is not implemented yet")

    # Get evaluators
    if not debug:
        model_acc = model.module
    else:
        model_acc = model

    if model_acc.semantic_algo is not None:
        sem_eval = model_acc.semantic_algo.evaluator
        num_classes = sem_eval.num_classes
    else:
        sem_eval, num_classes = None, 0
    if model_acc.instance_algo is not None and model_acc.semantic_algo is not None:
        panoptic_eval = model_acc.instance_algo.evaluator
        panoptic_eval.reset()
    else:
        panoptic_eval = None

    adapt_meters = {
        "losses": {
            "loss": AverageMeter(()),
            "depth_recon": AverageMeter(()),
            "depth_smth": AverageMeter(()),
            "flow_smth": AverageMeter(()),
            "flow_sparsity": AverageMeter(()),
            "semantic_source": AverageMeter(()),
            "semantic_cut_mixup": AverageMeter(()),
            "center_source": AverageMeter(()),
            "offset_source": AverageMeter(()),
        },
        "metrics": {
            # Depth
            "d_abs_rel": AverageMeter(()),
            "d_a1": AverageMeter(()),
            "d_a2": AverageMeter(()),
            "d_a3": AverageMeter(()),
            "d_rmse": AverageMeter(()),
            "d_rmse_log": AverageMeter(()),
            "d_sq_rel": AverageMeter(()),

            # Semantic
            "sem_conf": ConfusionMatrixMeter(num_classes),
            "sem_miou": ConstantMeter(()),
            "sem_miou_pixels": ConstantMeter(()),
            "sem_conf_interval": ConfusionMatrixMeter(num_classes),
            "sem_miou_interval": ConstantMeter(()),
            "sem_miou_pixels_interval": ConstantMeter(()),

            # Panoptic
            "p_pq": ConstantMeter(()),
            "p_sq": ConstantMeter(()),
            "p_rq": ConstantMeter(()),
            "p_stuff_pq": ConstantMeter(()),
            "p_stuff_sq": ConstantMeter(()),
            "p_stuff_rq": ConstantMeter(()),
            "p_things_pq": ConstantMeter(()),
            "p_things_sq": ConstantMeter(()),
            "p_things_rq": ConstantMeter(())
        }
    }
    # Class-wise depth statistics
    if do_class_wise_depth_stats:
        for i in range(dataloader.dataset.num_classes):
            adapt_meters["metrics"][f"d_abs_rel_c{i}"] = AverageMeter(())
            adapt_meters["metrics"][f"d_a1_c{i}"] = AverageMeter(())
            adapt_meters["metrics"][f"d_a2_c{i}"] = AverageMeter(())
            adapt_meters["metrics"][f"d_a3_c{i}"] = AverageMeter(())
            adapt_meters["metrics"][f"d_rmse_c{i}"] = AverageMeter(())
            adapt_meters["metrics"][f"d_rmse_log_c{i}"] = AverageMeter(())
            adapt_meters["metrics"][f"d_sq_rel_c{i}"] = AverageMeter(())

    # Create EMA clone
    if ema_modules:
        ema_model = CodepsNet.ema_model(model)
        if not debug:
            ema_model_acc = ema_model.module
        else:
            ema_model_acc = ema_model
    else:
        ema_model, ema_model_acc = None, None

    # Main adaptation loop
    for it, sample in enumerate(dataloader):
        if "target" not in sample:
            print(it)
            continue

        # Move input data to cuda
        sample_cuda = train_utils.dict_to_cuda(sample, device)

        # -------------------------------------------------------- #

        # Evaluate model on current sample before adaptation step
        if ema_model is not None:
            ema_model.eval()
            with torch.no_grad():
                _, results_eval, stats_eval = ema_model(sample_cuda["target"], "eval",
                                                        rgb_frame_offsets,
                                                        do_panoptic_fusion,
                                                        do_class_wise_depth_stats)

                for key, item in sample_cuda.items():
                    if key.endswith("mixup"):
                        _, results_plabel_eval, _ = ema_model(item, "eval", rgb_frame_offsets,
                                                              False, False)
                        if "semantic_head" in ema_modules or "instance_head" in ema_modules:
                            item["semantic_pred_tgt_ema"] = results_plabel_eval[
                                "semantic"].detach().clone()
                        if "instance_head" in ema_modules:
                            item["center_pred_tgt_ema"] = results_plabel_eval[
                                "center"].detach().clone()
                            item["offset_pred_tgt_ema"] = results_plabel_eval[
                                "offset"].detach().clone()
        else:
            model.eval()
            with torch.no_grad():
                _, results_eval, stats_eval = model(sample_cuda["target"], "eval",
                                                    rgb_frame_offsets, do_panoptic_fusion,
                                                    do_class_wise_depth_stats)

        # Complete the predictions for the mix-up strategy
        if "instance_head" not in ema_modules:
            for key, item in sample_cuda.items():
                if key.endswith("mixup"):
                    _, results_plabel_eval, _ = model(item, "eval", rgb_frame_offsets, False, False)
                    item["center_pred_tgt_ema"] = results_plabel_eval["center"].detach().clone()
                    item["offset_pred_tgt_ema"] = results_plabel_eval["offset"].detach().clone()

        for stat_name, stat_value in stats_eval.items():
            if stat_value is not None and stat_name in adapt_meters["metrics"]:
                adapt_meters["metrics"][stat_name].update(stat_value.cpu())
                if stat_name == "sem_conf":
                    adapt_meters["metrics"]["sem_conf_interval"].update(stat_value.cpu())
        # Semantics computation: Reduce confusion matrix to non-ignored classes only
        if sem_eval is not None and "semantic" in sample_cuda["target"]:
            # Take the confusion matrix from the last and all previous frames
            sem_conf_mat_filtered = sem_eval.filter_sem_conf_mat(
                adapt_meters["metrics"]["sem_conf"].sum, device, debug)
            # Classes that are not covered in the ground truth should not be considered
            indices_with_gt = sem_conf_mat_filtered.sum(dim=1) != 0
            sem_miou_score = sem_eval.compute_sem_miou(sem_conf_mat_filtered)[
                indices_with_gt].mean()
            sem_miou_score_pixels = sem_eval.compute_sem_miou(sem_conf_mat_filtered, True)
            adapt_meters["metrics"]["sem_miou"].update(sem_miou_score)
            adapt_meters["metrics"]["sem_miou_pixels"].update(sem_miou_score_pixels)

            # Take the confusion matrix from the last and all previous frames
            sem_conf_mat_interval_filtered = sem_eval.filter_sem_conf_mat(
                adapt_meters["metrics"]["sem_conf_interval"].sum, device, debug)
            # Classes that are not covered in the ground truth should not be considered
            indices_with_gt = sem_conf_mat_interval_filtered.sum(dim=1) != 0
            sem_miou_score = sem_eval.compute_sem_miou(sem_conf_mat_interval_filtered)[
                indices_with_gt].mean()
            sem_miou_score_pixels = sem_eval.compute_sem_miou(sem_conf_mat_interval_filtered, True)
            adapt_meters["metrics"]["sem_miou_interval"].update(sem_miou_score)
            adapt_meters["metrics"]["sem_miou_pixels_interval"].update(sem_miou_score_pixels)
        else:
            sem_conf_mat_filtered = None
        # Panoptics computation
        if do_panoptic_fusion and "semantic" in sample_cuda["target"]:
            # Based on the last and all previous frames
            semantic_gt = sample_cuda["target"].get("semantic_eval", sample_cuda["target"]["semantic"])
            pan_img_gt, _ = model_acc.instance_algo.panoptic_fusion(
                semantic_gt,
                sample_cuda["target"]["center"],
                sample_cuda["target"]["offset"])
            panoptic_eval.update(pan_img_gt, results_eval["panoptic"])
            panoptic_scores = panoptic_eval.evaluate()
            adapt_meters["metrics"]["p_pq"].update(torch.tensor(panoptic_scores["All"]["pq"]))
            adapt_meters["metrics"]["p_sq"].update(torch.tensor(panoptic_scores["All"]["sq"]))
            adapt_meters["metrics"]["p_rq"].update(torch.tensor(panoptic_scores["All"]["rq"]))
            adapt_meters["metrics"]["p_stuff_pq"].update(
                torch.tensor(panoptic_scores["Stuff"]["pq"]))
            adapt_meters["metrics"]["p_stuff_sq"].update(
                torch.tensor(panoptic_scores["Stuff"]["sq"]))
            adapt_meters["metrics"]["p_stuff_rq"].update(
                torch.tensor(panoptic_scores["Stuff"]["rq"]))
            adapt_meters["metrics"]["p_things_pq"].update(
                torch.tensor(panoptic_scores["Things"]["pq"]))
            adapt_meters["metrics"]["p_things_sq"].update(
                torch.tensor(panoptic_scores["Things"]["sq"]))
            adapt_meters["metrics"]["p_things_rq"].update(
                torch.tensor(panoptic_scores["Things"]["rq"]))

        # -------------------------------------------------------- #

        # Now, do the adaptation step
        if not disable_adaptation:
            model.train()
            for module in frozen_modules:
                module_ = getattr(model if debug else model.module, module, None)
                if module_ is not None:
                    module_.eval()  # Disable batch norm

            step_losses = {}
            for _ in range(adapation_steps):
                optimizer.zero_grad()
                losses, results, _, sample_cuda = model(sample_cuda, "adapt", rgb_frame_offsets,
                                                        do_panoptic_fusion=True)

                # Disable losses by setting values to None if there is no corresponding weight
                for loss_name in losses.keys():
                    if loss_weights[loss_name] is None:
                        losses[loss_name] = None

                if not step_losses:
                    step_losses = losses
                else:
                    step_losses = {k: v + step_losses[k] for k, v in losses.items() if
                                   v is not None}

                losses = OrderedDict(
                    (k, v.mean()) for k, v in losses.items() if v is not None and v.requires_grad)
                if losses:
                    losses["loss"] = sum(loss_weights[loss_name] * loss
                                         for loss_name, loss in losses.items())

                    # Increment the optimizer and backpropagate the gradients
                    losses["loss"].backward()
                    optimizer.step()

            # Update the EMA model
            if ema_model_acc is not None:
                # Use the EMA filter for the ema modules
                ema_model_acc.update_weights(model_acc, ema_modules, ema_alpha)
                # Otherwise, use the weights of the normal model updated with backprop
                ema_model_acc.update_weights(model_acc, ema_modules_inverted, 0)

            # Update the loss meters with the average loss over the adaptation steps
            for loss_name, loss_value in step_losses.items():
                if loss_value is not None:
                    adapt_meters["losses"][loss_name].update(loss_value.cpu() / adapation_steps)

            # Add adaptation (online) data to replay buffer
            dataloader.dataset.add(sample["index"].item(), results["image_features"])

        # -------------------------------------------------------- #

        # Log
        if (it + 1) % log_step_interval == 0:
            if wandb_summary is not None:
                # Log the last value (batch = True)
                logging.log_iter(adapt_meters["metrics"], None, it, len(dataloader), None, None,
                                 batch=True)
                logging.log_wandb(wandb_summary, "adapt", adapt_meters["losses"],
                                  adapt_meters["metrics"], None, True, it, None, it)
                logging.log_wandb_depth_class_v2(wandb_summary, "adapt", adapt_meters["metrics"],
                                                 True, it, config.dataset.remove_classes,
                                                 config.dataset.label_mode)
                wandb_vis_dict = {}
                max_vis_count = 5
                wandb_vis_dict, max_vis_count = gen_visualizations(sample_cuda,
                                                                   results_eval, wandb_vis_dict,
                                                                   img_scale, rgb_mean, rgb_std,
                                                                   max_vis_count,
                                                                   config.dataset.remove_classes,
                                                                   config.dataset.label_mode)
                if sem_conf_mat_filtered is not None:
                    # Add the visualization of the confusion matrix of the entire dataset.
                    wandb_vis_dict["conf_mat"] = \
                        plot_confusion_matrix(sem_conf_mat_filtered,
                                              config.dataset.remove_classes,
                                              config.dataset.label_mode)
                logging.log_wandb_images("adapt/batch", wandb_vis_dict, wandb_summary, it)
            adapt_meters["metrics"]["sem_conf_interval"] = ConfusionMatrixMeter(num_classes)

        # -------------------------------------------------------- #

        # Periodically save the snapshot
        # if (it + 1) % 10 == 0 and not debug:
        #     snapshot_file = Path(
        #         __file__).parent.absolute() / "snapshots" / f"snapshot_{it + 1}.pth"
        #     snapshot_file.parent.mkdir(exist_ok=True, parents=True)
        #     logging.log_info("Saving snapshot to %s", snapshot_file, debug=debug)
        #     io_utils.save_checkpoint(str(snapshot_file), config, it, it,
        #                              model.module.get_state_dict())

    # -------------------------------------------------------- #

    # Log the average metrics over the adaptation iterations
    if wandb_summary is not None:
        logging.log_wandb(wandb_summary, "adapt", None, adapt_meters["metrics"], None, False,
                          len(dataloader), None, len(dataloader))
        logging.log_wandb_depth_class_v2(wandb_summary, "adapt", adapt_meters["metrics"], False,
                                         len(dataloader), config.dataset.remove_classes,
                                         config.dataset.label_mode)
        if sem_eval is not None:
            # Add the visualization of the confusion matrix of the entire dataset.
            sem_conf_mat_filtered = sem_eval.filter_sem_conf_mat(
                adapt_meters["metrics"]["sem_conf"].sum, device, debug)
            wandb_vis_dict = {"conf_mat": plot_confusion_matrix(sem_conf_mat_filtered,
                                                                config.dataset.remove_classes,
                                                                config.dataset.label_mode)}
            logging.log_wandb_images("adapt/total", wandb_vis_dict, wandb_summary, len(dataloader))

    # For continual learning
    # dataloader.dataset.save_state()

    # Return the adapted model
    if ema_model is not None:
        return ema_model
    return model


def main(args):
    # Set the random number seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load configuration
    config = io_utils.gen_config(args, adapt=True)
    if config.adapt.mode == "off":
        print("\033[91m" + "WARNING: disabled adaptation" + "\033[0m")

    # Initialize devices
    if not args.debug:
        device_id, device, rank, world_size = train_utils.init_device()
        assert world_size == 1
    else:
        print("\033[91m" + "ACTIVE DEBUG MODE" + "\033[0m")
        rank, world_size = 0, 1
        device_id, device = rank, torch.device(rank + 0)

    # Create directories
    if not args.debug:
        log_dir, run_dir, saved_models_dir = io_utils.create_run_directories(args, rank)
    else:
        log_dir, run_dir, saved_models_dir = None, None, None

    # Initialize logging
    if not args.debug:
        wandb_summary = train_utils.init_logging(args, log_dir=log_dir, run_dir=run_dir,
                                                 config=config, init_wandb=rank == 0)
    else:
        wandb_summary = None

    # Create dataloaders
    train_dataloader = adapt_utils.gen_train_dataloader(args, config, rank, world_size)
    val_dataloader = adapt_utils.gen_val_dataloader(args, config, rank, world_size)
    source_val_dataloader = adapt_utils.gen_source_val_dataloader(args, config, rank, world_size)

    # Create model
    model = gen_models(config, device, train_dataloader.dataset.stuff_classes,
                       train_dataloader.dataset.thing_classes,
                       train_dataloader.dataset.ignore_classes,
                       train_dataloader.dataset.target_data.label_mode,
                       adaptation_mode=True)
    if args.checkpoint:
        logging.log_info("Loading checkpoint from %s", args.checkpoint, debug=args.debug)
        modules = io_utils.make_modules_list(config)
        io_utils.resume_from_checkpoint(model, args.checkpoint, modules)

    # Freeze modules that will not be trained
    frozen_modules = []
    frozen_modules += ["backbone_po_depth"] if not config.adapt.model.backbone_po_depth else []
    frozen_modules += ["backbone_pose_sflow"] if not config.adapt.model.backbone_pose_sflow else []
    frozen_modules += ["depth_head"] if not config.adapt.model.depth else []
    frozen_modules += ["pose_head"] if not config.adapt.model.pose else []
    frozen_modules += ["flow_head"] if not config.adapt.model.sflow else []
    frozen_modules += ["semantic_head"] if not config.adapt.model.semantic else []
    frozen_modules += ["instance_head"] if not config.adapt.model.instance else []
    model = train_utils.freeze_modules(frozen_modules, model)

    # Initialize GPU stuff
    model = train_utils.model_to_cuda(config, args, model, device, device_id)

    # Create optimizer and scheduler
    optimizer = train_utils.gen_optimizer(config.train.optimizer, model)

    # Adapt the model
    logging.log_info(f"Starting adaptation mode: {config.adapt.mode}. "
                     f"With {len(train_dataloader.dataset)} samples and "
                     f"{config.train.nof_adaptation_steps} adaptation steps.", debug=args.debug)
    model = adapt(model, optimizer, train_dataloader, frozen_modules, config, device,
                  wandb_summary, args.debug)
    logging.log_info(f"Starting validation with {len(val_dataloader.dataset)} samples.",
                     debug=args.debug)
    validate(model, val_dataloader, config, 0, wandb_summary, device, len(train_dataloader),
             compute_loss=False, print_results=True, debug=args.debug)

    # Validate with respect to the original dataset
    logging.log_info("Starting source validation.", debug=args.debug)
    validate(model, source_val_dataloader, config, len(train_dataloader), wandb_summary, device,
             len(train_dataloader), wandb_panel="source_val", compute_loss=False,
             print_results=True, debug=args.debug)

    logging.log_info("End of adaptation script!", debug=args.debug)
    if wandb_summary is not None:
        wandb_summary.finish()


if __name__ == "__main__":
    parser.add_argument("--mode", default="adapt", help=argparse.SUPPRESS)
    main(parser.parse_args())
