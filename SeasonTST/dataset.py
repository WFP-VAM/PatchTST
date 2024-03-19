import numpy
import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class SeasonTST_Dataset(Dataset):
    def __init__(
        self,
        dataset: xr.Dataset,
        time_array,
        size=None,
        pixels_per_epoch=1,
        train_size=.33,
        val_size=.33,
        split="train",
        scale=True,
    ):
        """
        BUG FIXES:
        - SeasonTST_Dataset is currently instantiated separately for train, valid and test
        see PatchTST_self_supervised/src/data/datamodule.py:44.
        This means train, valid and test are done on different pixels!


        TODO:
        Add validation checks for dataset argument:
        - expects spatial coordinates to be 'latitude', 'longitude'
        - expects dimension order to be time, lat , lon

        """

        if size is None:
            self.seq_len = 30
            self.label_len = 10
            self.pred_len = 10
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert split in ["train", "val", "test"]
        self.split = split
        self.train_size = train_size
        self.val_size = val_size
        self.scale = scale
        self.dataset = dataset
        self.time_array = time_array
        self.pixels_per_epoch = pixels_per_epoch
        self.features = list(dataset.data_vars.keys())
        self.set_split_data_length()

        self.initialize_data_for_epoch()

    def set_split_data_length(self):
        self.train_n = int(self.train_size * len(self.time_array))
        self.val_n = int(self.val_size * len(self.time_array))
        self.test_n = len(self.time_array) - self.val_n - self.train_n
        if self.split == "train":
            self.data_length = self.train_n
        elif self.split == "val":
            self.data_length = self.val_n
        else:
            self.data_length = self.test_n
        self.items_per_pixel = self.data_length - self.seq_len - self.pred_len + 1

    def initialize_data_for_epoch(self):
        # Randomly select a pixel
        lat, lon = self.select_random_pixel()
        print(
            f"(lat, lon) selected for {self.split}:",
            (self.lat, self.lon),
        )

        # while np.isnan(self.ndvi_xarray.isel(latitude=lat, longitude=lon, time=0).band.values) or np.isnan(self.rfh_xarray.isel(latitude=lat, longitude=lon, time=0).band.values):
        #     lat, lon = self.select_random_pixel()
        # Generate DataFrame for the selected pixel
        self.dataframe = self.generate_pixel_dataframe(lat, lon)

        # Read and split data
        self.__read_data__()

    def select_random_pixel(self):

        # TODO: add a check for pixels in the ocean
        lat = np.random.randint(0, self.dataset.latitude.shape[0])
        lon = np.random.randint(0, self.dataset.longitude.shape[0])

        self.lat = self.dataset.latitude.values[lat]
        self.lon = self.dataset.longitude.values[lon]
        return lat, lon

    def generate_pixel_dataframe(self, lat, lon):
        # Extract pd.DataFrame for a lat, lon indices combination
        df = pd.concat(
            [
                pd.DataFrame(
                    self.dataset.get(var_name)[:, lat, lon], columns=[var_name]
                )
                for var_name in self.dataset.data_vars
            ],
            axis=1,
        )

        df["time"] = self.time_array
        df.set_index("time", inplace=True)

        return df

    def __read_data__(self):
        df = self.dataframe.copy()

        if self.scale:
            self.scaler = StandardScaler()
            df[self.features] = self.scaler.fit_transform(df[self.features])


        if self.split == "train":
            self.data = df.iloc[:self.train_n]
            print(self.data.index.min(), self.data.index.max())
        elif self.split == "val":
            self.data = df.iloc[self.train_n : self.train_n + self.val_n]
            print(self.data.index.min(), self.data.index.max())
        else:
            self.data = df.iloc[self.train_n + self.val_n :]
            print(self.data.index.min(), self.data.index.max())

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data.iloc[s_begin:s_end][self.features].values
        seq_y = self.data.iloc[r_begin:r_end][self.features].values

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
            seq_y, dtype=torch.float32
        )

    def __len__(self):
        return self.pixels_per_epoch * self.items_per_pixel

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data
