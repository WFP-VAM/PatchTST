import json
from types import SimpleNamespace
import logging
import torch
import xarray as xr
from torch.utils.data import Dataset

from PatchTST_self_supervised.src.callback.patch_mask import PatchMaskCB
from PatchTST_self_supervised.src.callback.transforms import RevInCB
from PatchTST_self_supervised.src.data.datamodule import DataLoaders
from PatchTST_self_supervised.src.learner import Learner, transfer_weights
from PatchTST_self_supervised.src.models.patchTST import PatchTST
from SeasonTST.dataset import SeasonTST_Dataset


def get_dls(
    config_obj: SimpleNamespace,
    dataset_class: Dataset,
    dataset: xr.Dataset,
    mask: xr.DataArray,
):
    size = [config_obj.sequence_length, 0, config_obj.prediction_length]
    dls = DataLoaders(
        datasetCls=dataset_class,
        dataset_kwargs={
            "dataset": dataset,
            "mask": mask,
            "size": size,
        },
        batch_size=config_obj.batch_size,
        workers=config_obj.num_workers,
        prefetch_factor=config_obj.prefetch_factor,
    )

    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], config_obj.sequence_length
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls


def get_model(config, headtype="pretrain", weights_path=None, exclude_head=True):
    stride = config.stride
    # get number of patches
    num_patch = (
        max(config.sequence_length, config.patch_len) - config.patch_len
    ) // stride + 1

    # get model
    model = PatchTST(
        c_in=config.c_in,
        target_dim=config.prediction_length,
        patch_len=config.patch_len,
        stride=stride,
        num_patch=num_patch,
        n_layers=16,  # number of Transformer layers
        n_heads=64,  # number of Transformer heads
        d_model=128,  # Transformer d_model
        shared_embedding=True,
        d_ff=512,  # Tranformer MLP dimension
        dropout=2e-1,  # Transformer dropout
        head_dropout=2e-1,  # head dropout
        act="relu",
        head_type=headtype,
        res_attention=False,
    )
    # print out the model size
    print(
        "number of model params",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    if weights_path is not None:
        logging.info(f"Loading weights from {weights_path}")
        model = transfer_weights(weights_path, model, exclude_head)

    return model


def find_lr(config_obj, dls):
    """
    # This method typically involves training the model for a few epochs with a range of learning rates and recording
    the loss at each step. The learning rate that gives the fastest decrease in loss is considered optimal or
    near-optimal for the training process.

    :param config_obj:
    :return:
    """

    model = get_model(config_obj)
    # get loss
    loss_func = torch.nn.MSELoss(reduction="mean")
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if config_obj.revin else []
    cbs += [
        PatchMaskCB(
            patch_len=config_obj.patch_len,
            stride=config_obj.stride,
            mask_ratio=config_obj.mask_ratio,
            mask_value=-99,
        )
    ]

    # define learner
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=config_obj.lr,
        cbs=cbs,
    )
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print("suggested_lr", suggested_lr)
    return suggested_lr


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


def load_data():

    # Load dataset. Ensure it has no nans
    #PREFIX = "https://data.earthobservation.vam.wfp.org/public-share/"
    PREFIX = "/s3/scratch/public-share/"
    data = xr.open_zarr(PREFIX + "patchtst/Africa_data.zarr")
    data = data.sel(
        longitude=slice(9, 12), latitude=slice(-1, -3), time=slice("2003-01-01", None)
    )
    # downselect to only every 5 pixels
    data = data.thin({"latitude": 5, "longitude": 5})
    logging.info(f"Dataset dimensions: {data.dims}")

    # just one pixel
    data = data.isel(longitude=slice(0, 2), latitude=slice(0, 2))

    data = data.where(data.notnull(), -99)
    data = data.drop_vars("spatial_ref")
    data = data.transpose("time", "latitude", "longitude")

    # create ocean mask
    mask = data.sel(time=data.time.values[-1]).where(
        (data.sel(time=data.time.values[-1]) == -99), 0
    )
    mask = mask.drop_duplicates(dim="longitude")
    mask = sum([mask[v] for v in list(mask.keys())])
    mask = mask.where(mask == 0, -99)
    mask = mask == -99  # Make boolean
    mask = mask.compute()

    return data, mask
