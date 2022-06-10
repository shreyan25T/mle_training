# Mle_Training


[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)                 
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/) 

 # Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

- Linear regression
- Decision Tree
- Random Forest

## Installation:

### Prerequisites:

Prerequisite dependencies are stored in `deploy/conda/linux_cpu/py39.yml`. To setup the conda environment:

`$ conda env create -n test_env python`

`$ conda activate test-env`

### Setup:

Command to extract all the files required for package installation inside a newly created folder named
'Housing-1.0.0.tar.gz'
`tar -xvf dist/ Housing-1.0.0.tar.gz`

Go inside the folder:
`$ cd example_pkg`

Make log folder:
`$ mkdir logs`

For editable install:
`$ pip install -e .`

## Run code:

### To download and process data:

`$ python mypackage/housing_price/ingest_data.py -r data/raw/ -p data/processed/ --log-path logs/log.txt `

### To train the models:

`$ python mypackage/housing_price/train.py -d data/processed/housing_train.csv -m artifacts/ --log-path logs/log.txt `

### To score trained models:

`$ python mypackage/housing_price/score.py -d data/processed/housing_test.csv -m artifacts/ --log-path logs/log.txt `

### Note:

You can get information on command line arguments for each of the above scripts using `-h` or `--help`. For example:

`$ python src/housing_price/train.py --help`

## Steps performed:

- We prepared and cleaned the data.
- We checked and imputed missing values.
- Features are generated and the variables are checked for correlation.
- Multiple sampling techinuqies are evaluated. The data set is split into train and test.
- All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Testing

For running the unit cases run the following command
`python -m pytest ../tests`


