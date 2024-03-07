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

## For other datasets and settings

To train a dataset with name "dataset_name"
```bash
python src/main.py  --dataset dataset_name
```

To train a dataset for a baseline (e.g.full attention)

```bash
python src/main.py  --dataset dataset_name --model_name full
```

To train a dataset for a method with hyperparameters (epcochs and batch size)
python src/main.py  --dataset dataset_name --epochs 200 --batch_size 32



