import numpy
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Rain_Ndvi_Dataset(Dataset):
    def __init__(
        self,
        ndvi_array,
        rfh_array,
        time_array,
        lat_index,
        lon_index,
        size=None,
        split="train",
        scale=True,
    ):
        if size is None:
            self.seq_len = 30
            self.label_len = 10
            self.pred_len = 10
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert split in ["train", "val", "test"]
        self.split = split
        self.scale = scale
        # self.ndvi_xarray = ndvi_xarray
        # self.rfh_xarray = rfh_xarray
        self.ndvi_array = ndvi_array
        self.rfh_array = rfh_array
        self.time_array = time_array
        self.features = ["rfh", "ndvi"]
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
            print("(lat, lon) selected for test:", (lat, lon))

        self.dataframe = self.generate_pixel_dataframe(lat, lon)
        # Read and split data
        self.__read_data__()

    def select_random_pixel(self):
        lat = np.random.randint(0, self.rfh_xarray.latitude.shape[0])
        lon = np.random.randint(0, self.rfh_xarray.longitude.shape[0])
        return lat, lon

    def generate_pixel_dataframe(self, lat, lon):
        ndvi_df = pd.DataFrame(self.ndvi_array[:, lat, lon], columns=["band"])
        time_values = self.time_array
        ndvi_df = ndvi_df.reset_index()
        ndvi_df["time"] = time_values
        ndvi = ndvi_df[["time", "band"]]
        ndvi.rename(columns={"band": "ndvi"}, inplace=True)
        ndvi.set_index("time", inplace=True)

        rfh_df = pd.DataFrame(self.rfh_array[:, lat, lon], columns=["band"])
        time_values = self.time_array
        rfh_df = rfh_df.reset_index()
        rfh_df["time"] = time_values
        rfh = rfh_df[["time", "band"]]
        rfh.rename(columns={"band": "rfh"}, inplace=True)
        rfh.set_index("time", inplace=True)
        df = pd.concat([rfh, ndvi], axis=1)
        return df

    def __read_data__(self):
        df = self.dataframe.copy()

        if self.scale:
            self.scaler = StandardScaler()
            df[self.features] = self.scaler.fit_transform(df[self.features])

        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))

        if self.split == "train":
            self.data = df.iloc[:train_size]
        elif self.split == "val":
            self.data = df.iloc[train_size : train_size + val_size]
        else:
            self.data = df.iloc[train_size + val_size :]

    def __getitem__(self, index):
        # Choose a random start index for the sequence
        max_start_index = len(self.data) - self.seq_len - self.pred_len
        if max_start_index < 1:
            raise ValueError(
                "Dataset is too small for the specified sequence and prediction lengths."
            )
        random_start = np.random.randint(0, max_start_index)

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
