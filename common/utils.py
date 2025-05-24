import sys
import numpy as np
import json
import logging
import h5py
import random
import os
import torch
import glob
import yaml
from collections import OrderedDict
from datetime import datetime

yaml.Dumper.ignore_aliases = lambda *args : True

DEFAULT_RANDOM_SEED = 2022





def set_device(gpu=-1):
    import torch

    if gpu != -1 and torch.cuda.is_available():
        device = torch.device(
            "cuda:" + str(0)
        )  # already set env var in set logger function.
    else:
        device = torch.device("cpu")
    return device


def pprint(d, indent=0):
    d = sorted([(k, v) for k, v in d.items()], key=lambda x: x[0])
    for key, value in d:
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            pprint(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(round(value, 4)))


def load_hdf5(infile):
    logging.info("Loading hdf5 from {}".format(infile))
    with h5py.File(infile, "r") as f:
        return {key: f[key][:] for key in list(f.keys())}


def save_hdf5(outfile, arr_dict):
    logging.info("Saving hdf5 to {}".format(outfile))
    with h5py.File(outfile, "w") as f:
        for key in arr_dict.keys():
            f.create_dataset(key, data=arr_dict[key])


def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)


def load_json(infile):
    with open(infile, "r") as fr:
        return json.load(fr)


def update_from_nni_params(params, nni_params):
    if nni_params:
        params.update(nni_params)
    return params


def set_logger(args, is_prediction = False, log_file=None):
    args["model_root"] = os.path.join(args["result_dir"], args["expid"])
    args["benchmark_dir"] = os.path.join(args["model_root"], "benchmark_results")
    
    cur_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_file is None:
        if is_prediction == True:
            log_dir = os.path.join(
                args["model_root"], cur_time_str + "_prediction",
            )
        else:
            log_dir = os.path.join(
                args["model_root"], cur_time_str,
            )
        log_file = os.path.join(log_dir, "execution.log")
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    args["model_root"] = log_dir
    args["uptime"] = cur_time_str

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


    if args.get("device", -1) != -1:
        logging.info("Using device: cuda: {}".format(args["device"]))
        device = torch.device("cuda:{}".format(args["device"]))
    else:
        logging.info("Using device: cpu.")
        device = torch.device("cpu")

    
    return args, device
