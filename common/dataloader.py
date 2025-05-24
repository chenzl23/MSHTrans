import logging
import os
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import torch



def load_pkl(dataset_name, data_root):
    data_path = os.path.join(data_root, "{}.pkl".format(dataset_name))
    logging.info("Loading data from {}".format(data_path))
    with open(data_path, 'rb') as f:
        loaded_data = pickle.load(f)
    entities = loaded_data.keys()
    return loaded_data, entities

class sliding_window_dataset(Dataset):
    def __init__(self, window, timeseries, next_steps=0):
        self.window = window
        self.timeseries = timeseries
        self.next_steps = next_steps

    def __getitem__(self, index):
        if self.next_steps == 0:
            x = self.window[index]
            return_time_series_x = self.timeseries[index]
            return x, return_time_series_x
        else:
            x = self.window[index, 0 : -self.next_steps]
            y = self.window[index, -self.next_steps :]
            return_time_series_x = self.timeseries[index, 0 : -self.next_steps]
            return_time_series_y = self.timeseries[index, -self.next_steps :]
            return x, y, return_time_series_x, return_time_series_y

    def __len__(self):
        return len(self.window)


def get_dataloaders(
    train_data,
    test_data,
    train_time_series,
    test_time_series,
    valid_data=None,
    next_steps=0,
    batch_size=32,
    shuffle=True,
    num_workers=1,
):  

    train_loader = DataLoader(
        sliding_window_dataset(train_data, train_time_series, next_steps),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        sliding_window_dataset(test_data, test_time_series, next_steps),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    valid_loader = None
    return train_loader, valid_loader, test_loader