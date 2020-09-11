# Leader-based Multi-Scale Attention Deep Architecture for Person Re-identification
This repository contains the Pytorch implementation on Market-1501 for the paper ["Leader-based Multi-Scale Attention Deep Architecture for Person Re-identification"](http://epubs.surrey.ac.uk/852875/1/final_version.pdf)

## Framework

## Getting Started
### Prerequisites
* Python 3.6 or 3.7
* Pytorch >= 1.1.0

### Prepare data
Pleae download Market-1501 dataset and organize it as follows

    MuDeep_v2
        ├── dataset
        │      └─ Market-1501 # for Market-1501 dataset
        │             ├── bounding_box_train
        │             ├── bounding_box_test
        │             ├── query
        │
        ├── train.py
 
 ### Train
 In `config.py`, set configurations for training, including `MODEL_NAME`, `GPU_ID`, `PROJECT_FOLDER` or `BATCH_SIZE`
 ``` python
 __C.NAME = 'market'  # name your model, the log and model files will be created according to this name
 __C.GPU_ID = 0,1  
 __C.ROOT = PATH_TO_PROJECT_FOLDER  # all models and log files will be saved in this folder
 
 ...
 ```
 
 In `train.py`, using the following commend lines to train the model
 
 ``` python
 engine = MuDeep_v2(cfg)
 engine.train()
 ```
 
 ### Evaluate
 
 ## Results
