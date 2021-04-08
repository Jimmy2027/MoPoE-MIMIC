# MoPoE-MIMIC
This repository contains the code for the framework in Multimodal Generative Learning on the MIMIC-CXR Database (see [paper](https://openreview.net/forum?id=ZVqjoKVbYMl)).

It is based on the framework used in Generalized Multimodal ELBO (see [paper](https://openreview.net/forum?id=5Y21V0RDBV)). 

## Installation

```
git clone https://github.com/Jimmy2027/joint_elbo
cd joint_elbo
git checkout hendrik_mimic
path/to/conda/environment/bin/python -m pip install .
```

For development, install with: 
```
git clone https://github.com/Jimmy2027/joint_elbo
cd joint_elbo
git checkout hendrik_mimic
path/to/conda/environment/bin/python -m pip install -e .
```
to enable testing:
```
git clone https://github.com/Jimmy2027/joint_elbo
cd joint_elbo
git checkout hendrik_mimic
path/to/conda/environment/bin/python -m pip install -e .[test]
```
### Note
If pip throws an SSL Error, create first a new conda environment with `conda env create -f environment.yml`, and then install mimic using the steps above.
## Usage
Run the main training workflow with:
```
cd mimic
python main_mimic.py
```
A json config in `configs` can be used to give arguments to `main.py` with the flag `--config_path`. Note that the parameters in the config will be overwritten by the arguments passed through the command line.
```
cd mimic
python main_mimic.py --config_path path_to_my_json_config
```  
Otherwise an additional condition can be added in `mimic.utils.filehandling.get_config` so that the config is found automatically.

## Training the classifiers
See [here](https://github.com/Jimmy2027/joint_elbo/tree/distributed_training/mimic/networks/classifiers) for instructions on how to train the classifiers.
## Testing
run unittests with:
```
cd mimic
python -m pytest tests/
```
or more specifically:
```
cd mimic
python -m unittest tests/test_that_you_want_to_run.py
```
## Creating the tensor dataset
The tensor dataset can be created with the script `dataio/create_tensor_dataset.py`.
The creation of the tensor dataset consists of two steps. In a first step, the images of the original dataset are resized to a wanted size and stored as jpg in a folder. 
The first step is only executed if the folder of the resized images, or a zipped version of it is not found.

During the second step, the jpg images are read into a torch tensor and saved as such.

