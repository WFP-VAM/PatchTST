import logging
import multiprocessing
import time
import json
import numpy as np
import pandas as pd
import torch
import xarray as xr
import xbatcher
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset as TorchDataset


class TimeLatLonDataset(TorchDataset):
    """
    Subclass of Torch Dataset to accept any arbitrary xr.Dataset and creates batches using xbatcher.
    Together with shuffle=True, this achieves randomization of the extracted series across time and space.

    Ispired by:
    - https://earthmover.io/blog/cloud-native-dataloader
    - https://github.com/earth-mover/dataloader-demo

    """

    def __init__(
        self,
        dataset: xr.Dataset,
        mask: xr.DataArray,
        size=None,
        train_size=0.70,
        val_size=0.15,
        split="train",
        scale=True,
    ):

        if size is None:
            self.seq_len = 36
            self.label_len = 0
            self.pred_len = 9
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert split in ["train", "val", "test"]
        self.split = split

        """
        TODO: Add validation checks for dataset argument:
            - expects spatial coordinates to be 'latitude', 'longitude'
            - expects dimension order to be time, lat , lon
        """
        self.dataset = dataset
        self.mask = mask
        self.features = list(dataset.data_vars.keys())

        self.train_size = train_size
        self.val_size = val_size
        self.set_split_time_idxs()

        # TODO: StandardScaling is not currently implemented
        self.scale = scale

        # Create generator of batches of size extracted from the dataset
        self.set_batch_generator()

    def set_split_time_idxs(self):
        """
        self.dataset is split into train, val and test across time. This function identifies the relevant time_idxs
        for slicing
        """

        self.train_n = int(self.train_size * len(self.dataset.time))
        self.val_n = int(self.val_size * len(self.dataset.time))
        self.test_n = len(self.dataset.time) - self.val_n - self.train_n
        if self.split == "train":

            self.time_idxs = (None, self.train_n)
        elif self.split == "val":
            self.time_idxs = (self.train_n, self.train_n + self.val_n)
        else:
            self.time_idxs = (self.train_n + self.val_n, None)

    def get_split_dataset(self):
        return self.dataset.isel(time=slice(*self.time_idxs))

    def is_not_masked(self, selector: list[dict]):
        # Returns True if north west pixel of batch is not masked
        selector = selector[0]
        return not self.mask.isel(
            {"latitude": selector["latitude"], "longitude": selector["longitude"]}
        ).values[0][0]

    def set_batch_generator(self):

        data = self.get_split_dataset()

        series_len = self.seq_len + self.label_len + self.pred_len

        logging.info(f"Generating batches for {self.split} split")
        # For info: https://xbatcher.readthedocs.io/en/latest/demo.html
        batch_gen = data.batch.generator(
            input_dims={"time": series_len, "longitude": 1, "latitude": 1},
            input_overlap={"time": series_len - 1},
            preload_batch=False,
        )

        # Filter batch selectors based on mask
        logging.info(f"Filtering batch selectors based on mask for {self.split} split")
        selectors = [
            selector
            for idx, selector in batch_gen._batch_selectors.selectors.items()
            if self.is_not_masked(selector)
        ]

        # Re-index selectors and overwrite in batch_gen
        selectors = {idx: selector for idx, selector in enumerate(selectors)}
        batch_gen._batch_selectors.selectors = selectors

        logging.info(
            f"Completed generating {len(batch_gen)} batches for {self.split} split"
        )

        self.batch_gen = batch_gen

    def __len__(self):
        return len(self.batch_gen)

    def __getitem__(self, idx):
        t0 = time.time()
        logging.debug(
            json.dumps(
                {
                    "event": "get-batch start",
                    "time": t0,
                    "idx": idx,
                    "pid": multiprocessing.current_process().pid,
                }
            )
        )

        # load before stacking
        batch = self.batch_gen[idx].load()
        logging.debug(
            f"{batch.latitude.values}, {batch.longitude.values}, {batch.time.values[0]}"
        )

        # Stack to [time x var] shape
        stacked = (
            batch.to_stacked_array(
                new_dim="var", sample_dims=("time", "latitude", "longitude")
            )
            .squeeze()
            .transpose("time", "var", ...)
        )

        # Split time axis into input and ouput
        # TODO This split assumes  self.label_len is always 0.
        x = stacked.isel(time=slice(None, self.seq_len))
        y = stacked.isel(time=slice(self.seq_len, None))

        t1 = time.time()
        logging.debug(
            json.dumps(
                {
                    "event": "get-batch end",
                    "time": t1,
                    "idx": idx,
                    "pid": multiprocessing.current_process().pid,
                    "duration": t1 - t0,
                }
            )
        )
        return torch.tensor(x.data, dtype=torch.float32), torch.tensor(
            y.data, dtype=torch.float32
        )


class SeasonTST_Dataset(TimeLatLonDataset):
    def __init__(self, scaling_factors: dict = None, **kw_args):
        # TODO Include required parent positional arguments in __init__
        if scaling_factors is None:
            scaling_factors = {
                "mean": {
                    "ET0": 5.805,
                    "LST_SMOOTHED_5KM": 38.99,
                    "NDVI_SMOOTHED_5KM": 0.3417,
                    "RFH_DEKAD": 16.89,
                    "SOIL_MOIST": 0.09117,
                },
                "std": {
                    "ET0": 2.308,
                    "LST_SMOOTHED_5KM": 9.665,
                    "NDVI_SMOOTHED_5KM": 0.2629,
                    "RFH_DEKAD": 29.48,
                    "SOIL_MOIST": 0.1336,
                },
            }
        self.scaling_factors = scaling_factors
        super().__init__(**kw_args)
