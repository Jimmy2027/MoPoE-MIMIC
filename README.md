# joint_elbo

## Installation
For development, install with: 
```
git clone https://github.com/Jimmy2027/joint_elbo
cd joint_elbo
path/to/conda/environment/bin/python -m pip install -e .
```
or to enable testing:
```
git clone https://github.com/Jimmy2027/joint_elbo
cd joint_elbo
path/to/conda/environment/bin/python -m pip install -e .[test]
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