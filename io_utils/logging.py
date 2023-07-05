import logging
from collections import OrderedDict
from math import log10
from os import path
from typing import Any, Dict, List, Optional

import wandb
from torch import distributed

from datasets import get_labels
from eval.meters import AverageMeter, ConstantMeter

_NAME = "CoDEPS"


def _current_total_formatter(current, total):
    width = int(log10(total)) + 1
    return ("[{:" + str(width) + "}/{:" + str(width) + "}]").format(current, total)


def init(log_dir, name, file_logging=True):
    logger = logging.getLogger(_NAME)
    logger.setLevel(logging.DEBUG)

    # Set console logging
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # Setup file logging
    if file_logging:
        file_handler = logging.FileHandler(path.join(log_dir, name + ".log"), mode="w")
        file_formatter = logging.Formatter(fmt="%(levelname).1s %(asctime)s %(message)s",
                                           datefmt="%y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)


def get_logger():
    return logging.getLogger(_NAME)


def iteration(epoch,
              num_epochs,
              step,
              num_steps,
              values,
              multiple_lines=False):
    logger = get_logger()

    # Build message and write summary
    msg = _current_total_formatter(epoch, num_epochs)
    if step is not None:
        msg += " " + _current_total_formatter(step, num_steps)
    for k, v in values.items():
        if isinstance(v, AverageMeter):
            msg += "\n" if multiple_lines else "" + f"\t{k}={v.value.item():.5f} " \
                                                    f"({v.mean.item():.5f})"
        elif isinstance(v, ConstantMeter):
            msg += "\n" if multiple_lines else "" + f"\t{k}={v.value.item():.5f}"
        else:
            msg += "\n" if multiple_lines else "" + f"\t{k}={v:.5f}"

    # Write log
    logger.info(msg)


def log_info(msg, *args, **kwargs):
    if kwargs.get("debug", False):
        print(msg % args)
    else:
        kwargs.pop("debug", None)
        msg = f"[{distributed.get_rank()}] {msg}"
        get_logger().info(msg, *args, **kwargs)


def log_iter(meters, metrics, epoch, num_epochs, curr_iter, num_iters, lr=None, batch=True):
    iou = ["sem_conf", "sem_conf_interval"]
    log_entries = []

    # Ignore class-wise depth metrics
    ignore = []
    for i in range(20):
        ignore += [f"d_abs_rel_c{i}", f"d_a1_c{i}", f"d_a2_c{i}", f"d_a3_c{i}", f"d_rmse_c{i}",
                   f"d_rmse_log_c{i}", f"d_sq_rel_c{i}"]

    if lr is not None:
        log_entries = [("lr", lr)]

    meters_keys = list(meters.keys())
    meters_keys.sort()
    for meter_key in meters_keys:
        if meter_key in ignore:
            continue
        if meter_key in iou:
            log_value = meters[meter_key].iou.mean().item()
        else:
            if not batch:
                log_value = meters[meter_key].mean.item()
            else:
                log_value = meters[meter_key]
        log_key = meter_key
        log_entries.append((log_key, log_value))

    if metrics is not None:
        metrics_keys = list(metrics.keys())
        metrics_keys.sort()
        for metric_key in metrics_keys:
            if metric_key in ignore:
                continue
            log_key = metric_key
            if not batch:
                log_value = metrics[log_key].mean.item()
            else:
                log_value = metrics[log_key]
            log_entries.append((log_key, log_value))

    iteration(epoch + 1, num_epochs, curr_iter, num_iters, OrderedDict(log_entries))


# -------------------------------------------------------- #
# weights & biases


def log_wandb(
        wandb_summary,
        mode: str,
        loss_meters: Optional[Dict[str, AverageMeter]],
        metrics_meters: Optional[Dict[str, AverageMeter]],
        time_meters: Optional[Dict[str, AverageMeter]],
        batch: bool,
        epoch: int,
        learning_rate: Optional[float],
        global_step: int,
):
    ignore = ["sem_conf", "sem_conf_interval"]
    suffix = "batch" if batch else "total"

    # Ignore class-wise depth metrics
    for i in range(20):
        ignore += [f"d_abs_rel_c{i}", f"d_a1_c{i}", f"d_a2_c{i}", f"d_a3_c{i}", f"d_rmse_c{i}",
                   f"d_rmse_log_c{i}", f"d_sq_rel_c{i}"]

    def _get_log_value(meters, key):
        if key in ignore:
            log_value = None
            # log_value = meters[key].iou.mean().item()
        else:
            if batch:
                log_value = meters[key].value.item()
            else:
                log_value = meters[key].mean.item()
        return log_value

    log_dict = {}
    if loss_meters is not None:
        for meter_key in loss_meters.keys():
            if meter_key in ignore:
                continue
            log_key = f"{mode}_losses/{suffix}/{meter_key}"
            log_dict[log_key] = _get_log_value(loss_meters, meter_key)
    if metrics_meters is not None:
        for meter_key in metrics_meters.keys():
            if meter_key in ignore:
                continue
            log_key = f"{mode}_metrics/{suffix}/{meter_key}"
            log_dict[log_key] = _get_log_value(metrics_meters, meter_key)
    if time_meters is not None:
        for meter_key in time_meters.keys():
            if meter_key in ignore:
                continue
            log_key = f"misc/{mode}/{suffix}/{meter_key}"
            log_dict[log_key] = _get_log_value(time_meters, meter_key)

    # Other stuff
    log_dict["misc/epoch"] = epoch
    if learning_rate is not None:
        log_dict["misc/lr"] = learning_rate

    wandb_summary.log(log_dict, step=global_step)


def log_wandb_images(mode: str, wandb_vis_dict: Dict[str, Any], wandb_summary, global_step: int):
    for key, value in wandb_vis_dict.items():
        if key == "conf_mat" and mode != "train":
            images = [wandb.Image(vis) for vis in value] if isinstance(value, list) else \
                wandb.Image(value)
        else:
            images = [wandb.Image(vis.permute(1, 2, 0).detach().cpu().numpy()) for vis in value]
        wandb_summary.log({f"visual/{mode}/{key}": images}, step=global_step)


def log_wandb_table_panoptic(mode: str, panoptic_scores: dict, wandb_summary, global_step: int,
                             remove_classes: List[int], label_mode: str):
    dataset_labels = get_labels(remove_classes, label_mode)
    id2name = {label.trainId: label.name for label in dataset_labels}

    columns = [""]
    pq = ["PQ"]
    sq = ["SQ"]
    rq = ["RQ"]
    for category, scores in panoptic_scores["per_class"].items():
        columns.append(id2name[category])
        pq.append(scores["pq"])
        sq.append(scores["sq"])
        rq.append(scores["rq"])
    rows = [pq, sq, rq]
    table = wandb.Table(data=rows, columns=columns)
    wandb_summary.log({f"{mode}_metrics/panoptic": table}, step=global_step)


def log_wandb_depth_class(
        wandb_summary,
        metrics_meters: Dict[str, AverageMeter],
        batch: bool,
        global_step: int,
        remove_classes: List[int],
        label_mode: str,
        initialize_buffer: bool = False,
):
    if initialize_buffer:
        log_wandb_depth_class.line_series = {}

    keys = []
    for i in range(20):
        keys += [f"d_abs_rel_c{i}", f"d_a1_c{i}", f"d_a2_c{i}", f"d_a3_c{i}", f"d_rmse_c{i}",
                   f"d_rmse_log_c{i}", f"d_sq_rel_c{i}"]

    labels = get_labels(remove_classes, label_mode)
    class_labels = {label.trainId: label.name for label in labels}

    def _get_log_value(meters, key):
        if batch:
            log_value = meters[key].value.item()
        else:
            log_value = meters[key].mean.item()
        return log_value

    for meter_key in metrics_meters.keys():
        if meter_key in keys:
            value = _get_log_value(metrics_meters, meter_key)
            metric, c = meter_key.split("_c")
            c = class_labels[int(c)]
            if metric not in log_wandb_depth_class.line_series :
                log_wandb_depth_class.line_series[metric] = {
                    "xs": [global_step],
                    "ys": {c: [value]},
                    "keys": [c]
                }
            elif c not in log_wandb_depth_class.line_series[metric]["keys"]:
                log_wandb_depth_class.line_series[metric]["ys"][c] = [value]
                log_wandb_depth_class.line_series[metric]["keys"].append(c)
            else:
                if global_step not in log_wandb_depth_class.line_series[metric]["xs"]:
                    log_wandb_depth_class.line_series[metric]["xs"].append(global_step)
                log_wandb_depth_class.line_series[metric]["ys"][c].append(value)

    log_dict = {}
    for metric, value in log_wandb_depth_class.line_series.items():
        log_dict[metric] = wandb.plot.line_series(
            xs = value["xs"],
            ys = [value["ys"][c] for c in value["keys"]],
            keys= value["keys"],
            title=metric
        )
    wandb_summary.log(log_dict, step=global_step)
log_wandb_depth_class.line_series = {}

def log_wandb_depth_class_v2(
        wandb_summary,
        mode: str,
        metrics_meters: Optional[Dict[str, AverageMeter]],
        batch: bool,
        global_step: int,
        remove_classes: List[int],
        label_mode: str,
):
    suffix = "batch" if batch else "total"

    keys = []
    for i in range(20):
        # keys += [f"d_abs_rel_c{i}", f"d_a1_c{i}", f"d_a2_c{i}", f"d_a3_c{i}", f"d_rmse_c{i}",
        #            f"d_rmse_log_c{i}", f"d_sq_rel_c{i}"]
        keys += [f"d_rmse_c{i}"]

    labels = get_labels(remove_classes, label_mode)
    class_labels = {label.trainId: label.name for label in labels}

    def _get_log_value(meters, key):
        if batch:
            log_value = meters[key].value.item()
        else:
            log_value = meters[key].mean.item()
        return log_value

    log_dict = {}
    if metrics_meters is not None:
        for meter_key in metrics_meters.keys():
            if meter_key not in keys:
                continue
            metric, c = meter_key.split("_c")
            log_key = f"{mode}_depth/{suffix}/{metric}_{int(c):02}_{class_labels[int(c)]}"
            log_dict[log_key] = _get_log_value(metrics_meters, meter_key)

    wandb_summary.log(log_dict, step=global_step)
