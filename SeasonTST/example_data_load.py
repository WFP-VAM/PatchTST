from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from SeasonTST.dataset import SeasonTST_Dataset
from SeasonTST.utils import get_dataloaders

config = {
    "c_in": 2,  # number of variables
    "sequence_length": 36,
    "prediction_length": 9,
    "patch_len": 5,  # Length of the patch
    "stride": 5,
    "revin": 1,  # reversible instance normalization
    "mask_ratio": 0.4,  # masking ratio for the input
    "lr": 1e-3,
    "batch_size": 128,
    "num_workers": 0,
    "n_epochs_pretrain": 2500,  # number of pre-training epochs
    "pretrained_model_id": 2500,  # id of the saved pretrained model
}

config_obj = SimpleNamespace(**config)

PREFIX = "https://data.earthobservation.vam.wfp.org/public-share/"
standardized_indicators = xr.open_zarr(PREFIX + "CDI/standardized_indicators_AFv2")


# Creates train valid and test datasets for one epoch. Notice that they are in different locations!
dls = get_dataloaders(config_obj, SeasonTST_Dataset, standardized_indicators)

# Extract a batch
train_features, train_labels = next(iter(dls.test))


train_features[1, :, 1]

train_features[2, :, 1]
