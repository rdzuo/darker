# DARKER
The code of DARKER.
## Setup
```bash
pip install -r requirements.txt
```


We use the UEA Download the datasets, download the raw, preprocess, weights data from .
One dataset BasicMotions is provided for running an example.

Put raw data in dir:
```bash
data/raw/
```
Put the preprocessed data in dir:
```bash
data/preprocess/
```

Put the weights data in dir:

```bash
data/preprocess_learned/
```


install requirement:
```bash
pip install -r requirements.txt
```


## Run an example
To run an example:
```bash
cd src
python main.py --dataset BasicMotions
```

## For other dataset and settings

To train a dataset with name "dataset_name"
```bash
python src/main.py --data_path data/raw/  --dataset dataset_name
```
