# Leader-based Multi-Scale Attention Deep Architecture for Person Re-identification
This repository contains the Pytorch implementation on Market-1501 for the paper ["Leader-based Multi-Scale Attention Deep Architecture for Person Re-identification"](http://epubs.surrey.ac.uk/852875/1/final_version.pdf)

## Prerequisites
* Python 3.6 or 3.7
* Pytorch >= 1.1.0

## Prepare data
Pleae download Market-1501 dataset and organize it as follows

# For Market-1501 dataset
    MuDeep_v2
        ├── dataset
        │      └─ Market-1501
        │             ├── bounding_box_train
        │             ├── bounding_box_test
        │             ├── query
        │
        ├── train.py
