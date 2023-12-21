import xarray as xr
from rechunker import rechunk

PATH = "./"

def rechunk_batch(ds, target_chunks, max_mem, intermediate_store, target_store):
    rechunk_plan = rechunk(ds, target_chunks, max_mem, target_store, temp_store=intermediate_store)
    rechunk_plan.execute()

ds_full = xr.open_zarr("s3://wfp-ops-userdata/public-share/rfh_world.zarr")


# Define the new chunk sizes
new_time_chunk_size = -1
new_latitude_chunk_size = 50
new_longitude_chunk_size = 50
target_chunks = {'time':
                 new_time_chunk_size, 'latitude': new_latitude_chunk_size, 'longitude': new_longitude_chunk_size}

max_mem = "12GB"

time_step_size = 30  # Define batch size
num_batches = ds_full.dims['time'] // time_step_size

for i in range(num_batches):
    print("iteration", i)
    start = i * time_step_size
    end = start + time_step_size
    ds_batch = ds_full.isel(time=slice(start, end))

    intermediate_store = PATH + f'rfh_intermediate_store_batch_{i}.zarr'
    target_store = PATH + f'rfh_target_store_batch_{i}.zarr'
    
    rechunk_batch(ds_batch, target_chunks, max_mem, intermediate_store, target_store)

# Code to combine all the rechunked batches goes here
