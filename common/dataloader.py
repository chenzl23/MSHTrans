import logging
import os
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset





def load_dataset(
    data_root,
    entities,
    valid_ratio,
    dim,
    test_label_postfix,
    test_postfix,
    train_postfix,
    nan_value=0,
    nrows=None,
):
    """
    use_dim: dimension used in multivariate timeseries
    """
    logging.info("Loading data from {}".format(data_root))

    data = defaultdict(dict)
    total_train_len, total_valid_len, total_test_len = 0, 0, 0
    for dataname in entities:
        with open(
            os.path.join(data_root, "{}_{}".format(dataname, train_postfix)), "rb"
        ) as f:
            train = pickle.load(f)
            train=train.to_numpy()[:, :dim]
            # train.reshape((-1, dim))[0:nrows, :]
            if valid_ratio > 0:
                split_idx = int(len(train) * valid_ratio)
                train, valid = train[:-split_idx], train[-split_idx:]
                data[dataname]["valid"] = np.nan_to_num(valid, nan_value)
                total_valid_len += len(valid)
            data[dataname]["train"] = np.nan_to_num(train, nan_value)
            total_train_len += len(train)
        with open(
            os.path.join(data_root, "{}_{}".format(dataname, test_postfix)), "rb"
        ) as f:
            test = pickle.load(f)
            test = test.to_numpy()[:, :dim]
            # test.reshape((-1, dim))[0:nrows, :]
            data[dataname]["test"] = np.nan_to_num(test, nan_value)
            total_test_len += len(test)
        with open(
            os.path.join(data_root, "{}_{}".format(dataname, test_label_postfix)), "rb"
        ) as f:
            data[dataname]["test_label"] = pickle.load(f)
            data[dataname]["test_label"]=data[dataname]["test_label"] .to_numpy()
            data[dataname]["test_label"].reshape(-1)[0:nrows]
            data[dataname]["test_label"] = data[dataname]["test_label"].reshape(-1)

    logging.info("Loading {} entities done.".format(len(entities)))
    logging.info(
        "Train/Valid/Test: {}/{}/{} lines.".format(
            total_train_len, total_valid_len, total_test_len
        )
    )

    return data


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