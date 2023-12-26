# PatchTST (ICLR 2023)

### This is an offical implementation of PatchTST: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730). 

:triangular_flag_on_post: Our model has been included in [GluonTS](https://github.com/awslabs/gluonts). Special thanks to the contributor @[kashif](https://github.com/kashif)!

:triangular_flag_on_post: Our model has been included in [NeuralForecast](https://github.com/Nixtla/neuralforecast). Special thanks to the contributor @[kdgutier](https://github.com/kdgutier) and @[cchallu](https://github.com/cchallu)!

:triangular_flag_on_post: Our model has been included in [timeseriesAI(tsai)](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/15_PatchTST_a_new_transformer_for_LTSF.ipynb). Special thanks to the contributor @[oguiza](https://github.com/oguiza)!

We offer a video that provides a concise overview of our paper for individuals seeking a rapid comprehension of its contents: https://www.youtube.com/watch?v=Z3-NrohddJw



## Key Designs

:star2: **Patching**: segmentation of time series into subseries-level patches which are served as input tokens to Transformer.

:star2: **Channel-independence**: each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/model.png)

## Results

### Supervised Learning

Compared with the best results that Transformer-based models can offer, PatchTST/64 achieves an overall **21.0%** reduction on MSE and **16.7%** reduction
on MAE, while PatchTST/42 attains a overall **20.2%** reduction on MSE and **16.4%** reduction on MAE. It also outperforms other non-Transformer-based models like DLinear.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table3.png)

### Self-supervised Learning

We do comparison with other supervised and self-supervised models, and self-supervised PatchTST is able to outperform all the baselines. 

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table4.png)

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table6.png)

We also test the capability of transfering the pre-trained model to downstream tasks.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table5.png)

## Efficiency on Long Look-back Windows

Our PatchTST consistently <ins>reduces the MSE scores as the look-back window increases</ins>, which confirms our model’s capability to learn from longer receptive field.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/varying_L.png)

## Getting Started

We seperate our codes for supervised learning and self-supervised learning into 2 folders: ```PatchTST_supervised``` and ```PatchTST_self_supervised```. Please choose the one that you want to work with.

### Supervised Learning

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/PatchTST```. The default model is PatchTST/42. For example, if you want to get the multivariate forecasting results for weather dataset, just run the following command, and you can open ```./result.txt``` to see the results once the training is done:
```
sh ./scripts/PatchTST/weather.sh
```

You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.). We also provide codes for the baseline models.

### Self-supervised Learning

1. Follow the first 2 steps above

2. Pre-training: The scirpt patchtst_pretrain.py is to train the PatchTST/64. To run the code with a single GPU on ettm1, just run the following command
```
python patchtst_pretrain.py --dset ettm1 --mask_ratio 0.4
```
The model will be saved to the saved_model folder for the downstream tasks. There are several other parameters can be set in the patchtst_pretrain.py script.
 
 3. Fine-tuning: The script patchtst_finetune.py is for fine-tuning step. Either linear_probing or fine-tune the entire network can be applied.
```
python patchtst_finetune.py --dset ettm1 --pretrained_model <model_name>
```

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai

## Contact

If you have any questions or concerns, please contact us: ynie@princeton.edu or nnguyen@us.ibm.com or submit an issue

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@inproceedings{Yuqietal-2023-PatchTST,
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author    = {Nie, Yuqi and
               H. Nguyen, Nam and
               Sinthong, Phanwadee and 
               Kalagnanam, Jayant},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

## RAM-C work

No major changes done on the authors code. we only add a class function `capture_embedding()` in `PatchTST/PatchTST_self_supervised/src/models/patchTST.py`. 


These are the new notebooks


* `prepare-data.ipynb`: A Jupyter Notebook 
    1. A notebook to preprocess data. First this code do a **Rechunking** for the ndvi_world.zarr and are saved as 'NDVI Rechunked/ndvi_target_store_batch_{i}.zarr' and   'RFJ Rechunked/rfh_target_store_batch_{i}.zarr'. Then combine the rechunked data into one xarray.dataset
    2. Compute LTA of RFH and NDVI
    
* `Discovering.ipynb`: A Jupyter Notebook -  presentation of PatchTST 
    1. A first test of preprocessing of authors's weather dataset to understand the train/test split
    2. A first supervised training and testing on univariate and multivariate time series
    
* `pretrain-random.py`:A script of Pretraining PatchTST on random pixels:
    1. Data Loading and Processing: The script loads NDVI data from an S3 bucket using xarray, and sets up configurations for the PatchTST model. It includes settings such as number of variables, sequence and prediction lengths, patch length, stride, learning rate, batch size, and number of epochs for pretraining.
    2. Custom Dataset Class: A Rain_Ndvi_Dataset class is defined, which extends torch.utils.data.Dataset. This class is responsible for handling the NDVI and RFH data, enabling operations like random pixel selection, scaling, splitting data into different sets, and preparing it for model training.

    3. Model Initialization and Configuration: The script includes functions to create, configure, and find the learning rate for the PatchTST model. This involves setting parameters like the number of patches, transformer layers, heads, model dimension, dropout rates, etc.

    4. Training and Validation: Key functions for pretraining the model are defined, including loss plotting, and callbacks for model training and saving. The script handles pretraining, including setting up data loaders, model configuration, loss functions, and training loops.

    5. Utility Functions: The script contains utility functions like plot_loss for visualizing training progress and get_dls for preparing data loaders for the training process.

    6. Operational Flow: The script demonstrates a workflow from data loading, preprocessing, model initialization, pretraining, to saving the trained model. It is structured to facilitate easy tracking of the training process and results.
    
* `finetune.ipynb`: A Jupyter Notebook - Fine-Tuning a Machine Learning Model:
    1. Data Loading and Preprocessing: This section of the notebook involves loading data , and performing initial data preprocessing steps. This might include data cleaning, normalization, feature extraction, and data transformation suitable for the model being fine-tuned.

    2. Model Loading and Configuration: In this part, a pre-trained  model is loaded, which could be based on frameworks like TensorFlow or PyTorch. The model's parameters and architecture are likely detailed, along with any necessary modifications for the fine-tuning process.

    3. Fine-Tuning Setup: This section includes configuring the fine-tuning process. It could involve setting hyperparameters, choosing a loss function, and selecting an optimizer. There might also be a focus on specific layers of the model that are targeted for fine-tuning.

    4. Holtwinters: A holtwinters predictor is implemented and used for benchmarking
    
    5. Benchmarking: this section compares the perfomance of PatchTST, Holtwinters and LTA
 
* `region_forecast.ipynb`:  A Script - Forecasting every pixel in a region

    The finetuned model is evaluated on every pixel of the region. For future uses we save the results of every pixel as `saved_models/denorm_Africa_Forecast//out_{lat}_{lon}.pt.`
    
    
* `region_forecast.ipynb`:  A Jupyter Notebook - Forecasting every pixel in a region

    Pixels forecasts `saved_models/denorm_Africa_Forecast//out_{lat}_{lon}.pt.` are loaded and used to compare the results of RFH and NDVI to LTA
    
* `compare embedding.ipynb`:  A Jupyter Notebook - Climatological embedding



    
    