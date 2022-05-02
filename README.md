pyIAAS: Software for Multivariate Time-series Forecasting Model Development with an Intelligent Automated NAS optimization framework

# What is pyIAAS?
pyIAAS is an open-source Python package that exploits one of the latest neural architecture search (NAS) frameworks, intelligent automated achitecture search (IAAS) (Yang et al. 2022), for the multivariate time-series forecasting (MTF) model development. The aim of the pyIAAS is to facilitate the future researchers in building a high-quality MTF model efficiently and effectively when considering the optimality of the network structures. pyIAAS contains four network modules as searching candidates, namley, convolutional neural networks (CNN), recurrent neural networks (RNN), long-short term memory (LSTM) neural networks and fully connected neural networks (FCN). During the searching process, reinforcement learning (RL) based meta-controllers are designed to sequentially make decisions to update the network structures. Please refer to [Yang et al. (2022)](https://arxiv.org/abs/2203.13563) for more detailed information about the IAAS framework.

# Installation
### Without GPU acceleration
In an environment 3.8+, pyIAAS can be installed via
```shell
pip install pyIAAS
```
### With GPU acceleration 
First, install the GPU version of PyTorch
```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 
```
For more information of  PyTorch, please check [this](https://pytorch.org/get-started/locally/)

Then, install the pyIAAS package
```shell
pip install pyIAAS
```
# Usage of pyIAAS

## Command line
Command line tool supplies two functions: search and predict

To search networks given a dataset (use the example in example/1. basic)
```shell
# Start the searching process of pyIAAS
pyIAAS search -c NASConfig.json -f VT_summer.csv -t RT_Demand
```
Note that the output directory should be empty!

After the searching process, the neural architectures are stored in the directory ```examples/1. basic/out_dir```

To predict with the best searched model

```shell
# Perform a prediction task given a dataset ```VT_summer_predict.csv```
pyIAAS predict -c NASConfig.json -f VT_summer_predict.csv -t RT_Demand -d out_dir -o prediction.csv
```

## Python API
To search the neural architectures, then perform a prediction task with the best searched model

```python
import pyIAAS

# set the basic information of the searching process
config_file = 'NASConfig.json'
input_file = 'VT_summer.csv'
target_name = 'RT_Demand'
test_ratio = 0.2  # the proportion of the test dataset in the whole dataset. It can be adjusted by users themself for specific tasks

# start the searching process
pyIAAS.run_search(config_file, input_file, target_name, test_ratio)

# set the basic information of a prediction task
config_file = 'NASConfig.json'
input_file = 'VT_summer.csv'
target_name = 'RT_Demand'
output_dir = 'out_dir'
prediction_file = 'VT_summer_predict.csv'

# perform the predicting task in VT_summer_predict.csv
pyIAAS.run_predict(config_file, input_file, target_name, output_dir, prediction_file)
```


## Output file explanations
- model.db: detailed records of all searched models
- each searched model contains:
  - prediction results of the test dataset
  - transformation table
  - model parameters of type ```.pth```
  - training loss curve 
  
## Customized module list
The modules used in the searching process is given in the configuration
file. The default configuration is 
```json
{
  "MaxLayers": 50,
  "timeLength": 168,
  "predictLength": 24,
  "IterationEachTime": 50,
  "MonitorIterations": 40,
  "NetPoolSize": 5,
  "BATCH_SIZE": 256,
  "EPISODE": 200,
  "GPU": true,
  "OUT_DIR": "out_dir",
  "Modules" : {
  "dense": {
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
  },
  "rnn":{
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
  },
  "lstm":{
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
  },
  "conv": {
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
  }
  }
}
```

The meaning of each term:
- MaxLayers : number of the maximum layers of the searched neural architecture
- timeLength : length of the input time-series data
- predictLength : prediction time length, e.g., two-hour ahead
- IterationEachTime : number of the training epochs at each searching episode
- MonitorIterations : epoch interval to print out the training information, e.g., training loss 
- NetPoolSize : size of the net pool
- BATCH_SIZE : batch size used in the training process
- EPISODE : searching times of the reinforcement learning actors
- GPU : use GPU or not; if true, the environment should use the GPU version of PyTorch
- OUT_DIR : output directory
- Modules : module information 
  - out_range : list of the output unit number 
  - editable : whether this module can be widened or not

## Extending new module
To create a new module, users should create a subclass of ```pyIAAS.model.module.NasModule```, and implement 
these reserved abstract functions

```python
from pyIAAS.model.module import NasModule
# this is a sample subclassing of NasModule  to
# illustrate how to customize a new module in the pyIAAS package
class NewModule(NasModule):
    @property
    def is_max_level(self):
        # return: True if this module reaches the max width level, False otherwise
        raise NotImplementedError()

    @property
    def next_level(self):
        # return: width of next level
        raise NotImplementedError()

    def init_param(self, input_shape):
        # initialize the parameters of this module
        self.on_param_end(input_shape)
        raise NotImplementedError()

    def identity_module(self, cfg, name, input_shape: tuple):
        # generate an identity mapping module
        raise NotImplementedError()

    def get_module_instance(self):
        # generate a model instance once and use it for the rest procedures
        raise NotImplementedError()

    @property
    def token(self):
        # return: string type token of this module
        raise NotImplementedError()

    def perform_wider_transformation_current(self):
        # generate a new wider module by the wider function-preserving transformation
        # this function is called by layer i and returns the realized random mapping to the IAAS framework for the next layer's wider transformation.
        raise NotImplementedError()

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        # generate a new wider module by the wider function-preserving transformation
        # this function is called by the layer i + 1
        raise NotImplementedError()
```

Add the module information to the configuration file as follows
```json
{
  "MaxLayers": 50,
  "timeLength": 168,
  "predictLength": 24,
  "IterationEachTime": 50,
  "MonitorIterations": 40,
  "NetPoolSize": 5,
  "BATCH_SIZE": 256,
  "EPISODE": 200,
  "GPU": true,
  "OUT_DIR": "out_dir",
  "Modules" : {
  "dense": {
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
    },
  "new_module": {
    "out_range": [4, 8, 12, 16, 24, 32, 48, 64, 80, 108, 144],
    "editable": true
    }
  }
}
```
Register this new module in the running code

```python
from pyIAAS import *
from new_module import NewModule
cfg = Config('NASConfig.json')
# register a new module to the global configuration
cfg.register_module('new_module', NewModule)
```

# Additional information
For more details of this algorithm, see [Yang et al. (2022)](https://arxiv.org/abs/2203.13563)

For more details of pyIAAS, see [doc](https://jasonlovedl.github.io/pyIAAS/pyIAAS.html)

Repository is [here](https://github.com/JasonloveDL/pyIAAS)