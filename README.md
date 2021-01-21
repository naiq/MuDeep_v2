# Leader-based Multi-Scale Attention Deep Architecture for Person Re-identification
This repository contains the Pytorch implementation for the paper ["Leader-based Multi-Scale Attention Deep Architecture for Person Re-identification"](http://epubs.surrey.ac.uk/852875/1/final_version.pdf)

## Framework
<img src='https://github.com/naiq/MuDeep_v2/blob/master/fig/framework.png' width=500 height=300 alt='framework'>

## Getting Started
### Prerequisites
* Python 3.6 or 3.7
* Pytorch >= 1.1.0
* tensorboardX

### Prepare data
Please download Market-1501 dataset and organize it as follows

    MuDeep_v2
        ├── dataset
        │      └─ Market-1501 # for Market-1501 dataset
        │             ├── bounding_box_train
        │             ├── bounding_box_test
        │             ├── query
        │
        ├── train.py
 
 ### How to train
 In `config.py`, set configurations for training, including `NAME`, `GPU_ID` and `ROOT`. You can keep others as default to reproduce the result.
 ``` python
 # example
 __C.NAME = 'market'  # name your model, the model files (.pkl) will be saved according to this name
 __C.GPU_ID = 0,1  
 __C.ROOT = '/home/qxl/work/mudeep_v2/'  # path to your project folder, all models and log files will be saved in this folder
 ...
 ```
 
 In `train.py`, using the following command lines to train the model
 
 ``` python
 # example
 engine = MuDeep_v2(cfg)
 engine.train()
 ```
 Once trained, the models and log file will be saved in `ROOT/model/NAME/` and `ROOT/log/NAME/` respectively.
 
 By default, we evaluate the model every 5 epochs, the results will be written in `ROOT/model/NAME/opt.txt`

 
 ### How to evaluate
 In `train.py`, using the following command lines to evaluate the model
 
 ``` python
 # example
 engine = MuDeep_v2(cfg)
 engine.test(model_path='/home/qxl/work/mudeep_v2/model/market',   # path to your model
             out_name='market_evaluate'  # name the output TXT file
            )
 ```
 
 ## Result
 ### Supervised Learning
 | **Name** | **Backbone** | **image size** | **mAP** | **Rank-1** | **Rank-5** | **Rank-10** | **url** |
 | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
 | market_v1 | ResNet-50 | 384 x 192 | 85.61 | 95.46 | 98.43 | 98.93 | [download](https://drive.google.com/file/d/1slg1i74LOkyPNRtfUWi8n6pExofOUL-d/view?usp=sharing) |
 | market_v2 | ResNet-50 | 384 x 128 | 85.39 | 95.25 | 98.22 | 98.99 | [download](https://drive.google.com/file/d/1_4LCwJHOUXBdwKVSF8TArWomyNq9-4Ao/view?usp=sharing) |
 
 ### Domain Generalization
 <table>
   <tr align="center">
      <th rowspan="2"> Name </th>
      <th rowspan="2"> Backbone </th>
      <th rowspan="2"> image size </th>
      <th colspan="3"> Rank-1/mAP </th>
   </tr>
   <tr align="center">
      <th> DukeMTMC-reID </th>
      <th> CUHK03-np Detected </th>
      <th> CUHK03-np Labeled </th>
   </tr>
   <tr align="center">
      <td> market_v1 </td>
      <td> ResNet-50 </td>
      <td> 384 x 192 </td>
      <td> 46.68/27.33 </td> <!-- duke -->
      <td> 10.64/8.36 </td> <!-- cuhk03np detect -->
      <td> 11.79/9.34 </td> <!-- cuhk03np label -->
   </tr>
   <tr align="center">
      <td> market_v2 </td>
      <td> ResNet-50 </td>
      <td> 384 x 128 </td>
      <td> 48.56/28.24 </td> <!-- duke -->
      <td> 12.00/10.08 </td> <!-- cuhk03np detect -->
      <td> 13.00/10.70 </td> <!-- cuhk03np label -->
   </tr>
</table>
 
 
 ## Citation
If you find this project useful in your research, please consider cite:

    @article{qian2019leader,
      title={Leader-based multi-scale attention deep architecture for person re-identification},
      author={Qian, Xuelin and Fu, Yanwei and Xiang, Tao and Jiang, Yu-Gang and Xue, Xiangyang},
      journal={IEEE transactions on pattern analysis and machine intelligence},
      volume={42},
      number={2},
      pages={371--385},
      year={2019},
      publisher={IEEE}
    }

## Contact

Any questions or discussion are welcome!

Xuelin Qian (<xlqian15@fudan.edu.cn>)
