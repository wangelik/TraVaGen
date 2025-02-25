# import table
import random as ra

import numpy as np
import torch
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.importer.xes import importer


def set_seed(seed: int = 0) -> None:
    """Utility function to seed PRNG algorithms

    Args:
        seed (int): seed number to set

    """

    # set seeds
    ra.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # use deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_variants(file: str) -> dict:
    """Utility function to import raw variants from XES event log

    Args:
        file (str): path and filename of XES log

    Returns:
        variants (dict): dict container of raw pm4py variants

    """

    # create event log from XES file
    event_log = importer.apply(file)
    # extract variants
    variants = variants_filter.get_variants(event_log)

    return variants


def unpack_variants(variants: dict) -> list[str]:
    """Utility function to unpack raw pm4py variants

    Args:
        variants (dict): raw pm4py variant container

    Returns:
        unpacked_variants (list): list of variant strings

    """

    # extract and unpack variants into list
    unpacked_variants = []
    for var in variants:
        unpacked_variants += [str(var)] * len(variants[var])

    return unpacked_variants


def v_print(input: str, verbosity: bool) -> None:
    """Verbosity-dependent stdout utility

    Args:
        input (str): formatted string to print
        verbosity (bool): flag to switch printing

    """

    if verbosity:
        print(input)
