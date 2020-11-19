# joint_elbo

## Installation
For development, install with: 
```
git clone https://github.com/Jimmy2027/joint_elbo
cd joint_elbo
path/to/conda/environment/bin/python -m pip install -e .
```
to enable testing:
```
git clone https://github.com/Jimmy2027/joint_elbo
cd joint_elbo
path/to/conda/environment/bin/python -m pip install -e .[test]
```

## Usage
Run main training workflow with:
```
cd mimic
python main_mimic.py
```
A json config in `configs` can be used to give arguments to `main.py` with `--config_path`
```
cd mimic
python main_mimic.py --config_path path_to_my_json_config
```  
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