import logging
import yaml
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import save_hdf5


BENCHMARK_DIR = "./benchmark_results"


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            {str(k): str(v) for k, v in obj.items()},
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def store_entity(
    args,
    entity,
    train_anomaly_score,
    test_anomaly_score,
    test_anomaly_label,
    time_tracker={},
):
    exp_folder = args["model_root"]
    entity_folder = os.path.join(exp_folder, entity)
    os.makedirs(entity_folder, exist_ok=True)

    # save params
    with open(os.path.join(exp_folder, "params.yaml"), "w") as fw:
        yaml.dump(args, fw)


    # save time
    json_pretty_dump(time_tracker, os.path.join(entity_folder, "time.json"))

    # save scores
    score_dict = {
        "test_anomaly_label": test_anomaly_label.astype(int),
        "test_anomaly_score": test_anomaly_score,
        "train_anomaly_score": train_anomaly_score
    }
    save_hdf5(os.path.join(entity_folder, f"score_{entity}.hdf5"), score_dict)

    logging.info(f"Saving results for {entity} done.")

