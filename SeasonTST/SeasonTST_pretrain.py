import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import xarray as xr
from dask.cache import Cache

# adding PatchTST to the system path (necessary for windows machines)
#import sys
#sys.path.insert(0, r"C:\Users\15133\Documents\WFP\PatchTST")

from PatchTST_self_supervised.src.callback.patch_mask import PatchMaskCB
from PatchTST_self_supervised.src.callback.tracking import SaveModelCB
from PatchTST_self_supervised.src.callback.transforms import RevInCB
from PatchTST_self_supervised.src.learner import Learner, transfer_weights
from SeasonTST.dataset import SeasonTST_Dataset
from SeasonTST.utils import find_lr, get_dls, get_model



# Set up Dask's cache. Will reduce repeat reads from zarr and speed up data loading
cache = Cache(1e10)  # 10gb cache
cache.register()

import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename=f'logs/{datetime.datetime.now().strftime("%Y_%m_%d_%I_%M")}_train.log',
    encoding="utf-8",
    level=logging.DEBUG,
)


def pretrain_func(save_pretrained_model, save_path, config_obj, model, dls, lr=0.001):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(save_path)

    # get dataloader
    # get model

    # pretrained_model_path = "saved_models/masked_patchtst/patchtst_pretrained_cw36_patch5_stride5_epochs-pretrain2000_mask0.4_model4.pth"
    # model = transfer_weights(pretrained_model_path, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction="mean")
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if config_obj.revin else []
    cbs += [
        PatchMaskCB(
            patch_len=config_obj.patch_len,
            stride=config_obj.stride,
            mask_ratio=config_obj.mask_ratio,
            mask_value=config_obj.mask_value
        ),
        SaveModelCB(monitor="valid_loss", fname=save_pretrained_model, path=save_path),
    ]
    # define learner
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        # metrics=[mse]
    )
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=config_obj.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder["train_loss"]
    valid_loss = learn.recorder["valid_loss"]
    df = pd.DataFrame(data={"train_loss": train_loss, "valid_loss": valid_loss})
    df.to_csv(
        save_path + save_pretrained_model + "_losses.csv",
        float_format="%.6f",
        index=False,
    )

    return train_loss, valid_loss

def load_config():
    # Config parameters
    config = {
        "c_in": 5,  # number of variables
        "sequence_length": 36,
        "prediction_length": 0,
        "patch_len": 4,  # Length of the patch
        "stride": 4,  # Minimum non-overlap between patchs. If equal to patch_len , patches will not overlap
        "revin": 1,  # reversible instance normalization
        "mask_ratio": 0.4,  # masking ratio for the input
        "mask_value": -99, # Value to assign to masked elements of data input
        "lr": 1e-3,
        "batch_size": 128,
        "prefetch_factor": 3,
        "num_workers": 6,
        "n_epochs_pretrain": 1,  # number of pre-training epochs
        "pretrained_model_id": 2500,  # id of the saved pretrained model
    }

    config_obj = SimpleNamespace(**config)
    return config_obj

def load_data():

    # Load dataset. Ensure it has no nans
    PREFIX = "https://data.earthobservation.vam.wfp.org/public-share/"
    data = xr.open_zarr(PREFIX + "patchtst/Africa_data.zarr")
    data = data.sel(
        longitude=slice(9, 12), latitude=slice(-1, -3), time=slice("2003-01-01", None)
    )
    # downselect to only every 5 pixels
    data = data.thin({"latitude": 5, "longitude": 5})
    logging.info(f"Dataset dimensions: {data.dims}")

    data = data.where(data.notnull(), -99)
    data = data.drop_vars("spatial_ref")
    data = data.transpose("time", "latitude", "longitude")

    # create ocean mask
    mask = data.sel(time=data.time.values[-1]).where((data.sel(time=data.time.values[-1]) == -99), 0)
    mask = mask.drop_duplicates(dim="longitude")
    mask = sum([mask[v] for v in list(mask.keys())])
    mask = mask.where(mask==0,-99)
    mask = mask == -99  # Make boolean

    return data, mask

def main():
    data, mask = load_data()
    config_obj = load_config()

    # Creates train valid and test datasets for one epoch. Notice that they are in different locations!
    dls = get_dls(config_obj, SeasonTST_Dataset, data, mask)

    model = get_model(config_obj)

    # suggested_lr = find_lr(config_obj, dls)
    # This is what I got on a small dataset. In case one wants to skip this for testing.
    suggested_lr = 0.00020565123083486514

    save_pretrained_model = (
        "patchtst_pretrained_cw"
        + str(config_obj.sequence_length)
        + "_patch"
        + str(config_obj.patch_len)
        + "_stride"
        + str(config_obj.stride)
        + "_epochs-pretrain"
        + str(config_obj.n_epochs_pretrain)
        + "_mask"
        + str(config_obj.mask_ratio)
        + "_model"
        + str(config_obj.pretrained_model_id)
    )
    save_path = "saved_models" + "/masked_patchtst/"
    pretrain_func(save_pretrained_model, save_path, config_obj, model, dls, suggested_lr)

    pretrained_model_name = save_path + save_pretrained_model + ".pth"

    model = transfer_weights(pretrained_model_name, model)

if __name__ == "__main__":
    main()