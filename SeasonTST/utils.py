from types import SimpleNamespace

import xarray as xr
from torch.utils.data import Dataset

from PatchTST_self_supervised.src.callback.patch_mask import PatchMaskCB
from PatchTST_self_supervised.src.callback.transforms import RevInCB
from PatchTST_self_supervised.src.data.datamodule import DataLoaders
from PatchTST_self_supervised.src.learner import Learner
from PatchTST_self_supervised.src.models.patchTST import PatchTST
from SeasonTST.dataset import SeasonTST_Dataset


def get_dataloaders(
    config_obj: SimpleNamespace,
    dataset_class: Dataset,
    dataset: xr.Dataset,
):
    size = [config_obj.sequence_length, 0, config_obj.prediction_length]
    dls = DataLoaders(
        datasetCls=dataset_class,
        dataset_kwargs={
            "dataset": dataset,
            "time_array": dataset.time.values,
            "size": size,
            "scale": True,
        },
        batch_size=config_obj.batch_size,
        workers=config_obj.num_workers,
    )

    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], config_obj.sequence_length
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls


def get_model(config, headtype="pretrain"):
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
    return model


def find_learning_rate(config_obj):
    """
    # This method typically involves training the model for a few epochs with a range of learning rates and recording
    the loss at each step. The learning rate that gives the fastest decrease in loss is considered optimal or
    near-optimal for the training process.

    :param config_obj:
    :return:
    """

    # get dataloader
    dls = get_dataloaders(config_obj, SeasonTST_Dataset)
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