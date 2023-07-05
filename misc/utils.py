from collections import OrderedDict

import torch
from torch import distributed as dist
from yacs.config import CfgNode

_VALID_TYPES = {tuple, list, str, int, float, bool, None}


def all_reduce_losses(losses):
    """Coalesced mean all reduce over a dictionary of 0-dimensional tensors"""
    names, values = [], []
    for k, v in losses.items():
        names.append(k)
        values.append(v)

    # Perform the actual coalesced all_reduce
    values = torch.cat([v.view(1) for v in values], dim=0)
    dist.all_reduce(values, dist.ReduceOp.SUM)
    values.div_(dist.get_world_size())
    values = torch.chunk(values, values.size(0), dim=0)

    # Reconstruct the dictionary
    return OrderedDict((k, v.view(())) for k, v in zip(names, values))


def convert_to_dict(cfg_node, key_list=None):
    """ Convert a config node to dictionary """

    key_list = [] if key_list is None else key_list

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(f"Key {'.'.join(key_list)} with value {type(cfg_node)} is not a valid type; "
                  f"valid types: {_VALID_TYPES}")
        return cfg_node

    cfg_dict = dict(cfg_node)
    for k, v in cfg_dict.items():
        cfg_dict[k] = convert_to_dict(v, key_list + [k])
    return cfg_dict
