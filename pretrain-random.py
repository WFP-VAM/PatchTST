import sys
sys.path.append("./PatchTST_self_supervised/")
from types import SimpleNamespace
import xarray as xr
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os


from PatchTST_self_supervised.src.models.patchTST import PatchTST
from PatchTST_self_supervised.src.learner import Learner, transfer_weights
from PatchTST_self_supervised.src.callback.tracking import *
from PatchTST_self_supervised.src.callback.patch_mask import *
from PatchTST_self_supervised.src.callback.transforms import *
from PatchTST_self_supervised.src.metrics import *
from PatchTST_self_supervised.src.basics import set_device
from PatchTST_self_supervised.datautils import *

from src.data.datamodule import DataLoaders



PATH = "./"

ds_full = xr.open_zarr("s3://wfp-ops-userdata/public-share/ndvi_world.zarr")

config = {
    'c_in' : 2 , #number of variables
    'sequence_length' : 36,
    'prediction_length' : 9,
    'patch_len' : 5, #Length of the patch
    'stride' : 5,
    
    'revin':1, #reversible instance normalization
    'mask_ratio':0.4, # masking ratio for the input
    
    'lr' : 1e-3,
    
    'batch_size':128, 
    'num_workers':0,
    
    'n_epochs_pretrain' : 2500, # number of pre-training epochs
    'pretrained_model_id': 2500, # id of the saved pretrained model

}

config_obj = SimpleNamespace(**config)
#
def plot_loss(train_loss, valid_loss, save_path):
    plt.clf()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss after Epoch {epoch}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'loss_plot_epoch.png'))
    plt.show()

LAT_LON = []


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
        lat = np.random.randint(0, self.rfh_xarray.latitude.shape[0])
        lon = np.random.randint(0, self.rfh_xarray.longitude.shape[0])
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



    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data



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

RFH = xr.concat(concatenated_ds_list, dim='time')


ndvi_array = NDVI.band.values
rfh_array = RFH.band.values
time_array = RFH.time.values

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



def get_model(config, headtype='pretrain'):
    stride  = config.stride    
    # get number of patches
    num_patch = (max(config.sequence_length, config.patch_len)-config.patch_len) // stride + 1    
    
    # get model
    model = PatchTST(c_in=config.c_in,
                target_dim=config.prediction_length,
                patch_len=config.patch_len,
                stride=stride,
                num_patch=num_patch,
                n_layers=16, #number of Transformer layers
                n_heads=64,#number of Transformer heads
                d_model=128, #Transformer d_model
                shared_embedding=True,
                d_ff=512, #Tranformer MLP dimension                       
                dropout=2e-1, #Transformer dropout
                head_dropout=2e-1, #head dropout
                act='relu',
                head_type=headtype,
                res_attention=False
                )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



print("PatchTST Model Created")
model = get_model(config_obj,'pretrain')


# +
def find_lr(config_obj):
    # get dataloader
    dls = get_dls(config_obj, Rain_Ndvi_Dataset)
    model = get_model(config_obj)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if config_obj.revin else []
    cbs += [PatchMaskCB(patch_len=config_obj.patch_len, stride=config_obj.stride, mask_ratio=config_obj.mask_ratio)]
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=config_obj.lr, 
                        cbs=cbs,
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr

suggested_lr=find_lr(config_obj)


# -

# This method typically involves training the model for a few epochs with a range of learning rates and recording the loss at each step. The learning rate that gives the fastest decrease in loss is considered optimal or near-optimal for the training process.

# ## Pretrain

print("Starting Pretraining ...")


def pretrain_func(save_pretrained_model, save_path, lr=suggested_lr):
    
     
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    print(save_path)

    # get dataloader
    dls = get_dls(config_obj, Rain_Ndvi_Dataset)
    # get model
    model = get_model(config_obj)
    # pretrained_model_path = "saved_models/masked_patchtst/patchtst_pretrained_cw36_patch5_stride5_epochs-pretrain2000_mask0.4_model4.pth"
    # model = transfer_weights(pretrained_model_path, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if config_obj.revin else []
    cbs += [
         PatchMaskCB(patch_len=config_obj.patch_len, stride=config_obj.stride, mask_ratio=config_obj.mask_ratio),
         SaveModelCB(monitor='valid_loss', fname=save_pretrained_model,                       
                        path=save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        #metrics=[mse]
                        )                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=config_obj.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(save_path + save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)
    

    return train_loss, valid_loss


save_pretrained_model = 'patchtst_pretrained_cw'+str(config_obj.sequence_length)+'_patch'+str(config_obj.patch_len) + '_stride'+str(config_obj.stride) + '_epochs-pretrain' + str(config_obj.n_epochs_pretrain) + '_mask' + str(config_obj.mask_ratio) + '_model' + str(config_obj.pretrained_model_id)
save_path = 'saved_models' + '/masked_patchtst/'
pretrain_func(save_pretrained_model, save_path)

pretrained_model_name= save_path + save_pretrained_model+".pth"

model = transfer_weights(pretrained_model_name, model)


# # # Finetune

# # +
# config = {
#     'c_in' : 2 , #number of variables
#     'sequence_length' : 36,
#     'prediction_length' : 9,
#     'patch_len' : 5, #Length of the patch
#     'stride' : 5,

#     'revin':1, #reversible instance normalization
#     'mask_ratio':0.4, # masking ratio for the input

#     'lr' : 1e-4,

#     'batch_size':64, 
#     'num_workers':0,

#     'n_epochs_pretrain' : 10, # number of pre-training epochs
#     'n_epochs_finetune' : 10, # number of pre-training epochs
#     'pretrained_model_id': 1, # id of the saved pretrained model


#     'save_finetuned_model': './finetuned',

#     'save_path' :  'saved_models' + '/masked_patchtst/'

# }

# config_obj = SimpleNamespace(**config)


# # -

# def get_model(args, head_type, weight_path=None):
#     """
#     c_in: number of variables
#     """
#     # get number of patches
#     num_patch = (max(args.sequence_length, args.patch_len)-args.patch_len) // args.stride + 1    
#     print('number of patches:', num_patch)

#     # get model
#     model = PatchTST(c_in=args.c_in,
#                 target_dim=args.prediction_length,
#                 patch_len=args.patch_len,
#                 stride=args.stride,
#                 num_patch=num_patch,
#                 n_layers=4,
#                 n_heads=16,
#                 d_model=128,
#                 shared_embedding=True,
#                 d_ff=512,                        
#                 dropout=2e-1, #Transformer dropout
#                 head_dropout=2e-1, #head dropout
#                 act='relu',
#                 head_type=head_type,
#                 res_attention=False
#                 )    
#     if weight_path: model = transfer_weights(weight_path, model)
#     # print out the model size
#     print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
#     return model


# weight_path="saved_models/masked_patchtst/patchtst_pretrained_cw36_patch5_stride5_epochs-pretrain10_mask0.4_model1.pth"


# def find_lr(args,head_type, weight_path):
#     # get dataloader
#     dls = get_dls(args, Rain_Ndvi_Dataset)    
#     model = get_model(args, head_type)
#     # transfer weight
#     # weight_path = args.save_path + args.pretrained_model + '.pth'
#     model = transfer_weights(weight_path, model)
#     # get loss
#     loss_func = torch.nn.MSELoss(reduction='mean')
#     # get callbacks
#     cbs = [RevInCB(dls.vars)] if args.revin else []
#     cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

#     # define learner
#     learn = Learner(dls, model, 
#                         loss_func, 
#                         lr=args.lr, 
#                         cbs=cbs,
#                         )                        
#     # fit the data to the model
#     suggested_lr = learn.lr_finder()
#     print('suggested_lr', suggested_lr)
#     return suggested_lr


# find_lr(config_obj,"prediction",weight_path)


# # +
# def save_recorders(args,learn):
#     train_loss = learn.recorder['train_loss']
#     valid_loss = learn.recorder['valid_loss']
#     df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
#     df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)


# def finetune_func(args,lr=config_obj.lr, weight_path= weight_path):
#     print('end-to-end finetuning')
#     # get dataloader
#     dls = get_dls(args, Rain_Ndvi_Dataset)
#     # get model 
#     model = get_model(args, head_type='prediction')
#     # transfer weight
#     # weight_path = args.pretrained_model + '.pth'
#     model = transfer_weights(weight_path, model)
#     # get loss
#     loss_func = torch.nn.MSELoss(reduction='mean')   
#     # get callbacks
#     cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
#     cbs += [
#          PatchCB(patch_len=args.patch_len, stride=args.stride),
#          SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
#         ]
#     # define learner
#     learn = Learner(dls, model, 
#                         loss_func, 
#                         lr=lr, 
#                         cbs=cbs,
#                         metrics=[mse]
#                         )                            
#     # fit the data to the model
#     #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
#     learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10)
#     save_recorders(args, learn)



# # -

# finetune_func(config_obj,lr=config_obj.lr, weight_path= weight_path)


# def test_func(args, weight_path):
#     # get dataloader
#     print('end-to-end finetuning')
#     # get dataloader
#     dls = get_dls(args, Rain_Ndvi_Dataset)
#     # get model 
#     model = get_model(args, head_type='prediction').to('cpu')


#     # get callbacks
#     cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
#     cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
#     learn = Learner(dls, model,cbs=cbs)
#     out  = learn.test(dls.test, weight_path=weight_path+'.pth', scores=[mse,mae])         # out: a list of [pred, targ, score]
#     print('score:', out[2])
#     # save results
#     pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
#     return out, dls

# out,dls = test_func(config_obj, config_obj.save_path + config_obj.save_finetuned_model)

# pred, targ, score = out
# gt = dls.test.dataset[0]

# # +
# import matplotlib.pyplot as plt
# import numpy as np

# def visualize(inputs, targets, outputs, idx, feature_names, seq_len):
#     # Select a sample from the batch to visualize

#     # Detach and move to CPU for visualization
#     inputs_sample = inputs.cpu().detach().numpy()
#     targets_sample = targets.cpu().detach().numpy()
#     outputs_sample = outputs.cpu().detach().numpy()

#     # Plotting
#     plt.figure(figsize=(12,3 ),  facecolor='none')

#     for i, feature_name in enumerate(feature_names):
#         print(i)
#         plt.subplot(1, len(feature_names), i + 1)

#         # Time axis
#         time_input = np.arange(seq_len)
#         time_future = np.arange(seq_len, seq_len + len(targets_sample))

#         # Plot input, target, and output
#         plt.plot(time_input, inputs_sample[:, i], label='Input')
#         plt.plot(time_future, targets_sample[:, i], label='Target')
#         plt.plot(time_future, outputs_sample[:, i], label='Prediction')

#         plt.title(f'Batch {idx+1}, {feature_name}')
#         plt.legend()

#     plt.tight_layout()
#     plt.show()



# # +
# pred, targ, score = out

# for idx in range(pred.shape[0]):

#     inputs = dls.test.dataset[idx][0]
#     targets= dls.test.dataset[idx][1]
#     outputs = torch.Tensor(pred[idx])

#     visualize(inputs, targets, outputs, idx, ['rain(mm)', 'ndvi'], config_obj.sequence_length)

# # -

# print("MSE = ", score[0])
# print("MAE = ", score[1])


