import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import xarray as xr
from dask.cache import Cache

from PatchTST_self_supervised.src.callback.patch_mask import PatchCB, ObservationMaskCB
from PatchTST_self_supervised.src.callback.tracking import SaveModelCB
from PatchTST_self_supervised.src.callback.transforms import RevInCB
from PatchTST_self_supervised.src.learner import Learner, transfer_weights
from SeasonTST.dataset import SeasonTST_Dataset
from SeasonTST.utils import find_lr, get_dls, get_model, load_data
from PatchTST_self_supervised.src.metrics import mse, mae

import logging
import datetime
import dask

#
# SETUP
#

# Set up Dask's cache. Will reduce repeat reads from zarr and speed up data loading
cache = Cache(1e10)  # 10gb cache
cache.register()
#dask.config.set(scheduler="threads", num_workers=2)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename=f'logs/{datetime.datetime.now().strftime("%Y_%m_%d_%I:%M")}_finetune.log',
    encoding="utf-8",
    level=logging.DEBUG,
)


#
#  FUNCTIONS
#


def finetune_func(learner, save_path, args, lr=0.001):
    print("end-to-end finetuning")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(save_path)
    # fit the data to the model and save
    learner.fine_tune(
        n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=args.freeze_epochs
    )
    save_recorders(learner, args)


def get_learner(args, dls, lr, model):
    """
    Learner set-up

    TRAINING
    - Input is [bs, seq_len, n_vars]
    - Before forward pass:
        - RevInCB normalized inputs
        - ObservationMaskCB masks random observations with fill value
        - PatchCB reshaped to [bs, num_patches, n_vars, patch_len]
    - Forward pass in: [bs, num_patches, n_vars, patch_len]; out: [bs, pred_len, n_vars]
    - After forward pass
        - RevInCB denormalized outputs
        - ObservationMaskCB custom loss function on outputs for just the masked values
        - Loss is therefore mean squared difference on denormalized masked values.
        - Will give more weight to variables with larger numerical range
    """



    # get loss
    loss_func = torch.nn.MSELoss(reduction="mean")
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
        #ObservationMaskCB(mask_ratio=0.2, mask_value=-99),
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(
            monitor="valid_loss", fname=args.save_finetuned_model, path=args.save_path
        ),
    ]
    # define learner
    learner = Learner(dls, model, loss_func, lr=lr, cbs=cbs, metrics=[mse])
    return learner


def save_recorders(learner, args):
    train_loss = learner.recorder["train_loss"]
    valid_loss = learner.recorder["valid_loss"]
    df = pd.DataFrame(data={"train_loss": train_loss, "valid_loss": valid_loss})
    df.to_csv(
        args.save_path + args.save_finetuned_model + "_losses.csv",
        float_format="%.6f",
        index=False,
    )


def test_func(weight_path, learner, args, dls):

    out = learner.test(
        dls.test, weight_path=weight_path, scores=[mse, mae]
    )  # out: a list of [pred, targ, score]
    print("score:", out[2])
    # save results
    pd.DataFrame(np.array(out[2]).reshape(1, -1), columns=["mse", "mae"]).to_csv(
        args.save_path + args.save_finetuned_model + "_acc.csv",
        float_format="%.6f",
        index=False,
    )
    return out


def load_config():

    # Config parameters
    # TODO maybe load from a JSON with a model key?
    config = {
        "c_in": 5,  # number of variables
        "sequence_length": 36,
        "prediction_length": 2,  # Sets both the dimension of y from the dataloader as well as the prediction head size
        "patch_len": 4,  # Length of the patch
        "stride": 4,  # Minimum non-overlap between patchs. If equal to patch_len , patches will not overlap
        "revin": 0,  # reversible instance normalization
        "mask_ratio": 0.4,  # masking ratio for the input
        "lr": 1e-3,
        "batch_size": 128,
        "num_workers": 6,
        "prefetch_factor": 3,
        "n_epochs_pretrain": 1,  # number of pre-training epochs,
        "freeze_epochs": 0,
        "n_epochs_finetune": 50,
        "pretrained_model_id": 2500,  # id of the saved pretrained model
        "save_finetuned_model": "./finetuned_d128",
        "save_path": "saved_models" + "/masked_patchtst/",
    }
    config_obj = SimpleNamespace(**config)

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
    pretrained_model_path = save_path + save_pretrained_model + ".pth"

    return config_obj, save_path, pretrained_model_path




#
# FINE TUNING STEPS
#
def main():
    data, mask = load_data()

    config_obj, save_path, pretrained_model_path = load_config()

    # This creates a new model using pretrained weights as a start
    # Use the finetuned checkpoint instead
    path = save_path + config_obj.save_finetuned_model[2:] + ".pth"
    #path = pretrained_model_path
    model = get_model(
        config_obj, headtype="prediction", weights_path=path, exclude_head=False
    )

    # Create dataloader
    dls = get_dls(config_obj, SeasonTST_Dataset, data, mask)


    #suggested_lr = find_lr(config_obj, dls)
    # This is what I got on a small dataset. In case one wants to skip this for testing.
    suggested_lr = 0.00017073526474706903
    print(suggested_lr)

    learner = get_learner(config_obj, dls, suggested_lr, model)

    # This function will save the model weights to config_obj.save_finetuned_model. ie will not overwrite the pretrained model.
    # However, there is currently no set-up to do finetuning from the result of a previous finetuning.
    finetune_func(learner, pretrained_model_path, config_obj, suggested_lr)


if __name__ == "__main__":
    main()
