import sys

from SeasonTST.dataset import SeasonTST_Dataset
from SeasonTST.utils import find_lr, get_dls, get_model

sys.path.append("../PatchTST_self_supervised/")
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import torch
import xarray as xr

from PatchTST_self_supervised.datautils import *
from PatchTST_self_supervised.src.callback.patch_mask import *
from PatchTST_self_supervised.src.callback.tracking import *
from PatchTST_self_supervised.src.callback.transforms import *
from PatchTST_self_supervised.src.learner import Learner, transfer_weights

PATH = "../"

ds_full = xr.open_zarr("s3://wfp-ops-userdata/public-share/ndvi_world.zarr")

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


#
def plot_loss(train_loss, valid_loss, save_path):
    plt.clf()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(valid_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss after Epoch {epoch}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"loss_plot_epoch.png"))
    plt.show()


LAT_LON = []

new_time_chunk_size = -1
new_latitude_chunk_size = 50
new_longitude_chunk_size = 50
target_chunks = {
    "time": new_time_chunk_size,
    "latitude": new_latitude_chunk_size,
    "longitude": new_longitude_chunk_size,
}

max_mem = "12GB"

time_step_size = 30  # Define batch size
num_batches = ds_full.dims["time"] // time_step_size


concatenated_ds_list = []

for i in range(num_batches):
    target_store = PATH + f"NDVI Rechunked/ndvi_target_store_batch_{i}.zarr"
    ds_rechunked = xr.open_zarr(target_store)
    concatenated_ds_list.append(ds_rechunked)

NDVI = xr.concat(concatenated_ds_list, dim="time")


concatenated_ds_list = []

for i in range(num_batches):
    target_store = PATH + f"RFH Rechunked/rfh_target_store_batch_{i}.zarr"
    ds_rechunked = xr.open_zarr(target_store)
    concatenated_ds_list.append(ds_rechunked)

RFH = xr.concat(concatenated_ds_list, dim="time")


ndvi_array = NDVI.band.values
rfh_array = RFH.band.values
time_array = RFH.time.values

print("PatchTST Model Created")
model = get_model(config_obj, "pretrain")


suggested_lr = find_lr(config_obj)


# -


# ## Pretrain

print("Starting Pretraining ...")


def pretrain_func(save_pretrained_model, save_path, lr=suggested_lr):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(save_path)

    # get dataloader
    dls = get_dls(config_obj, SeasonTST_Dataset)
    # get model
    model = get_model(config_obj)
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
pretrain_func(save_pretrained_model, save_path)

pretrained_model_name = save_path + save_pretrained_model + ".pth"

model = transfer_weights(pretrained_model_name, model)
