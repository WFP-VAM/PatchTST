# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: fix_hdc
#     language: python
#     name: conda-env-fix_hdc-py
# ---

# +
import sys
sys.path.append("./PatchTST_self_supervised/")
from types import SimpleNamespace
import xarray as xr
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np


import matplotlib.pyplot as plt

from PatchTST_self_supervised.src.models.patchTST import PatchTST
from PatchTST_self_supervised.src.learner import Learner, transfer_weights
from PatchTST_self_supervised.src.callback.tracking import *
from PatchTST_self_supervised.src.callback.patch_mask import *
from PatchTST_self_supervised.src.callback.transforms import *
from PatchTST_self_supervised.src.metrics import *
from PatchTST_self_supervised.src.basics import set_device
from PatchTST_self_supervised.datautils import *

from src.data.datamodule import DataLoaders

# -

# ## Dataset and Dataloader

# +
PATH = "./"

ds_full = xr.open_zarr("s3://wfp-ops-userdata/public-share/ndvi_world.zarr")

new_time_chunk_size = -1
new_latitude_chunk_size = 50
new_longitude_chunk_size = 50
target_chunks = {'time':
                 new_time_chunk_size, 'latitude': new_latitude_chunk_size, 'longitude': new_longitude_chunk_size}

max_mem = "12GB"

time_step_size = 30  # Define batch size
num_batches = ds_full.dims['time'] // time_step_size


concatenated_ds_list = []

for i in range(num_batches):
    target_store = PATH + f'NDVI Rechunked/ndvi_target_store_batch_{i}.zarr'
    ds_rechunked = xr.open_zarr(target_store)
    concatenated_ds_list.append(ds_rechunked)

NDVI = xr.concat(concatenated_ds_list, dim='time')


concatenated_ds_list = []

for i in range(num_batches):
    target_store = PATH + f'RFH Rechunked/rfh_target_store_batch_{i}.zarr'
    ds_rechunked = xr.open_zarr(target_store)
    concatenated_ds_list.append(ds_rechunked)

# Concatenate all datasets along the time dimension
RFH = xr.concat(concatenated_ds_list, dim='time')


# +
# NDVI_lta = NDVI.sel(time=slice("2003-02-01","2018-12-31"))
# NDVI_lta  = NDVI_lta .groupby(NDVI_lta.time.dt.strftime("%m-%d")).mean()
# NDVI_lta = NDVI_lta.sel(strftime=NDVI.time.dt.strftime("%m-%d"))

# RFH_lta = RFH.sel(time=slice("2003-02-01","2018-12-31"))
# RFH_lta  = RFH_lta .groupby(RFH_lta.time.dt.strftime("%m-%d")).mean()
# RFH_lta = RFH_lta.sel(strftime=RFH.time.dt.strftime("%m-%d"))

# +
# def get_LTA (lat,lon,index, seq_len, pred_len):
    
#     s_begin = index
#     s_end = s_begin + seq_len
#     r_begin = s_end 
#     r_end = r_begin + pred_len
        

#     ndvi_lta_prediction = NDVI_lta.isel(time=slice(r_begin,r_end), latitude=lat, longitude=lon).band.values
# #     rfh_lta_prediction = RFH_lta.isel(time=slice(r_begin,r_end), latitude=lat, longitude=lon).band.values
    
#     return ndvi_lta_prediction
 
# -

bbox_africa = [-17.314453,-34.957995,51.855469,13.667338]

# +
NDVI_africa =NDVI.sel(
    latitude=slice(bbox_africa[3], bbox_africa[1]),
    longitude=slice(bbox_africa[0], bbox_africa[2])
).isel(
    latitude=slice(None, None, 5), 
    longitude=slice(None, None, 5)
)


RFH_africa =RFH.sel(
    latitude=slice(bbox_africa[3], bbox_africa[1]),
    longitude=slice(bbox_africa[0], bbox_africa[2])
).isel(
    latitude=slice(None, None, 5), 
    longitude=slice(None, None, 5)
)


NDVI_africa.compute()
RFH_africa.compute()

# +
class Rain_Ndvi_Dataset(Dataset):
    def __init__(self, ndvi_array, rfh_array,time_array, lat_index,lon_index, size=None, split='train', scale=True):
        if size is None:
            self.seq_len = 30
            self.label_len = 10
            self.pred_len = 10
        else:
            self.seq_len, self.label_len, self.pred_len = size
            
        assert split in ['train', 'val', 'test']
        self.split = split
        self.scale = scale
        # self.ndvi_xarray = ndvi_xarray
        # self.rfh_xarray = rfh_xarray
        self.ndvi_array = ndvi_array
        self.rfh_array = rfh_array
        self.time_array = time_array
        self.features = ['rfh', 'ndvi']
        self.lat_index = lat_index
        self.lon_index = lon_index


        self.initialize_data_for_epoch()

    def initialize_data_for_epoch(self):
        # Randomly select a pixel
        lat, lon = self.select_random_pixel()
        
        # while np.isnan(self.ndvi_xarray.isel(latitude=lat, longitude=lon, time=0).band.values) or np.isnan(self.rfh_xarray.isel(latitude=lat, longitude=lon, time=0).band.values):
        #     lat, lon = self.select_random_pixel()
        # Generate DataFrame for the selected pixel
        if self.split == "test":
            print("(lat, lon) selected for test:",  (lat, lon))
            
        self.dataframe = self.generate_pixel_dataframe(lat, lon)
        # Read and split data
        self.__read_data__()

    def select_random_pixel(self):
        lat = self.lat_index
        lon = self.lon_index
        return lat, lon

    def generate_pixel_dataframe(self, lat, lon):
        ndvi_df = pd.DataFrame(self.ndvi_array[:,lat,lon], columns=['band'])
        time_values = self.time_array  
        ndvi_df = ndvi_df.reset_index()
        ndvi_df['time'] = time_values
        ndvi = ndvi_df[['time', 'band']]
        ndvi.rename(columns={'band': 'ndvi'}, inplace=True)
        ndvi.set_index('time', inplace=True)
        
        
        rfh_df = pd.DataFrame(self.rfh_array[:,lat,lon], columns=['band'])
        time_values = self.time_array  
        rfh_df = rfh_df.reset_index()
        rfh_df['time'] = time_values
        rfh = rfh_df[['time', 'band']]
        rfh.rename(columns={'band': 'rfh'}, inplace=True)
        rfh.set_index('time', inplace=True)
        df = pd.concat([rfh, ndvi], axis=1)
        return df

    def __read_data__(self):
        df = self.dataframe.copy()

        if self.scale:
            self.scaler = StandardScaler()
            df[self.features] = self.scaler.fit_transform(df[self.features])
            

        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))

        if self.split == 'train':
            self.data = df.iloc[:train_size]
        elif self.split == 'val':
            self.data = df.iloc[train_size:train_size + val_size]
        else:
            self.data = df.iloc[train_size + val_size:]

    def __getitem__(self, index):
        # Choose a random start index for the sequence
        max_start_index = len(self.data) - self.seq_len - self.pred_len
        if max_start_index < 1:
            raise ValueError("Dataset is too small for the specified sequence and prediction lengths.")
        random_start = np.random.randint(0, max_start_index)

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len
        

        seq_x = self.data.iloc[s_begin:s_end][self.features].values
        seq_y = self.data.iloc[r_begin:r_end][self.features].values


        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)


    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data


config = {
    'c_in' : 2 , #number of variables
    'sequence_length' : 36,
    'prediction_length' : 9,
    'patch_len' : 5, #Length of the patch
    'stride' : 5,
    
    'revin':1, #reversible instance normalization
    'mask_ratio':0.4, # masking ratio for the input
    
    'lr' : 3e-4,
    
    'batch_size':128, 
    'num_workers':0,
    
    'n_epochs_pretrain' : 200, # number of pre-training epochs
    'n_epochs_finetune' : 50, # number of pre-training epochs
    'pretrained_model_id': 1, # id of the saved pretrained model
    
    
    'save_finetuned_model': './finetuned_d128',
    
    'save_path' :  'saved_models' + '/masked_patchtst/'
    
}

config_obj = SimpleNamespace(**config)

def get_model(args, head_type, weight_path=None):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.sequence_length, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = PatchTST(c_in=args.c_in,
                target_dim=args.prediction_length,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=num_patch,
                n_layers=4, #number of Transformer layers
                n_heads=16,#number of Transformer heads
                d_model= 128, #128, #Transformer d_model
                shared_embedding=True,
                d_ff=512, #Tranformer MLP dimension                              
                dropout=2e-1, #Transformer dropout
                head_dropout=2e-1, #head dropout
                act='relu',
                head_type=head_type,
                res_attention=False
                )    
    if weight_path: model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

ndvi_array = NDVI_africa.band.values
rfh_array = RFH_africa.band.values
time_array = RFH_africa.time.values

def get_dls(config_obj, dataset_class,lat,lon):
    size = [config_obj.sequence_length, 0, config_obj.prediction_length]
    dls = DataLoaders(
            datasetCls=dataset_class,
            dataset_kwargs={
                'ndvi_array':ndvi_array , 
                'rfh_array':rfh_array ,
                'time_array':time_array ,
                'size':size,
                'scale':True,
                'lat_index': lat,
                'lon_index': lon
            },
            batch_size=config_obj.batch_size,
            workers=config_obj.num_workers,
            )

    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], config_obj.sequence_length
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls


def test_func(args, weight_path,lat,lon):
    # get dataloader
    print('end-to-end finetuning')
    # get dataloader
    dls = get_dls(args, Rain_Ndvi_Dataset,lat,lon)
    # get model 
    model = get_model(args, head_type='prediction').to('cpu')
    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=weight_path+'.pth', scores=[mse,mae])         # out: a list of [pred, targ, score]
    print('score:', out[2])
    # save results
    pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    return out, dls


for lat in range(NDVI_africa.latitude.shape[0]):
    for lon in range(NDVI_africa.longitude.shape[0]): 
        out,dls = test_func(config_obj, config_obj.save_path + config_obj.save_finetuned_model,lat,lon)
        mean,std= dls.test.dataset.scaler.mean_,dls.test.dataset.scaler.scale_
        pred, targ, score = out
        denorm_out=[pred*std+mean, targ*std+mean]
        torch.save(denorm_out, f"saved_models/denorm_Africa_Forecast/out_{lat}_{lon}.pt")


