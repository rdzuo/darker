# DARKER
The code of DARKER.
## Setup
```bash
pip install -r requirements.txt
```


We use the UEA Download the datasets, download the raw, preprocess, weights data from .

For a dataset named BasicMotions, put it as follows:

Put raw data from /raw/ in dir:
```bash
data/raw/BasicMotions
```
Put the preprocessed data from /preprocess/ in dir:
```bash
data/preprocess/BasicMotions
```

Put the weights data from /preprocess_learned/ in dir:

```bash
data/preprocess_learned/BasicMotions
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

```bash
python src/main.py  --dataset dataset_name --epochs 200 --batch_size 32
```



