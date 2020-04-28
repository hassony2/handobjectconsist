"""
Inspired from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
"""
import re

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch._six import container_abcs, string_classes
from meshreg.datasets.queries import BaseQueries, TransQueries


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def extend_collate(batch, extend_queries=None):
    """
    Collate function, duplicating the items in extend_queries along the
    first dimension so that they all have the same length
    """
    if extend_queries is None:
        extend_queries = []
    pop_queries = []
    for poppable_query in extend_queries:
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    for pop_query in pop_queries:
        max_size = max([sample[pop_query].shape[0] for sample in batch])
        for sample in batch:
            pop_value = sample[pop_query]
            # Repeat vertices so all have the same number
            pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
            sample[pop_query] = pop_value

    batch = default_collate(batch)
    return batch


def meshreg_collate(batch):
    """
    Collate function, duplicating the items in extend_queries along the
    first dimension so that they all have the same length.
    Typically applies to faces and vertices, which have different sizes
    depending on the object.
    """

    extend_queries = [
        TransQueries.OBJVERTS3D,
        BaseQueries.OBJVERTS3D,
        BaseQueries.OBJCANVERTS,
        BaseQueries.OBJCANROTVERTS,
        TransQueries.OBJCANROTVERTS,
        BaseQueries.OBJVERTS2D,
        BaseQueries.OBJVIS2D,
        TransQueries.OBJVERTS2D,
        BaseQueries.OBJFACES,
    ]

    pop_queries = []
    for poppable_query in extend_queries:
        if poppable_query in batch[0]:
            pop_queries.append(poppable_query)

    # Remove fields that don't have matching sizes
    for pop_query in pop_queries:
        max_size = max([sample[pop_query].shape[0] for sample in batch])
        for sample in batch:
            pop_value = sample[pop_query]
            # Repeat vertices so all have the same number
            pop_value = np.concatenate([pop_value] * int(max_size / pop_value.shape[0] + 1))[:max_size]
            sample[pop_query] = pop_value
    batch = default_collate(batch)
    return batch


def seq_extend_collate(seq, extend_queries=None):
    seq_len = len(seq[0])
    seqs = [[] for _ in range(seq_len)]
    for seq_idx in range(seq_len):
        for sample in seq:
            seqs[seq_idx].append(sample[seq_idx])
    return [extend_collate(sample, extend_queries) for sample in seqs]


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""

    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        # array of string classes and object
        if elem_type.__name__ == "ndarray" and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data).float()
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(default_convert(d) for d in data)
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data
