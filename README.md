# TADGCN: A Time-Aware Dynamic Graph Convolution Network for Long-term Traffic Flow Prediction

This is a TensorFlow implementation of TADGCN.

## Requirements

* TensorFlow-gpu==2.5.0
* numpy==1.19.5
* pandas==1.4.1
* networkx==2.6.3
* einops==0.3.2
* gensim==4.1.2
* pyyaml==5.4.1

## Dataset

The datasets used in our paper are collected by the Caltrans Performance Measurement System(PeMS). Please refer to [STSGCN (AAAI2020)](https://github.com/Davidham3/STSGCN) for the download url.

## Run

### Train
***
Before training this model, set `mode: train` in the corresponding yaml file in the `configs` folder, and execute the following command: 

    python main.py --dataset=PEMS04
    or
    python main.py --dataset=PEMS08

### Test
***
Before testing the TADGCN, you should modify `mode: test` and give the best model path. For example:

    best_path: ./experiments/PEMS08/2023-10-06-15-18-30/best_model

Subsequently, execute the following command to test the model:

    python main.py --dataset=PEMS08


## Citation

Please cite the following paper, if you find the repository or the paper useful.

    xxx

