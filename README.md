# Pytorch Sqlite Dataloader Example

This repo demonstrates using sqlite used as a backend for pytorch's `Torch.utils.data.Dataset` interface.

This code uses the netflix prize dataset as an example, but the same idea can be generally applied.

## Running

1. Download netflix prize data from:

https://www.kaggle.com/netflix-inc/netflix-prize-data

The scripts here assume that all .txt files are gzipped and stored under `./data/raw/netfix/`.

This the contents of `./data/raw/netflix/` should be:

```
README
combined_data_1.txt.gz
combined_data_2.txt.gz
combined_data_3.txt.gz
combined_data_4.txt.gz
movie_titles.csv
probe.txt.gz
qualifying.txt.gz
```

2. Construct a sqlite file by running `ingest_netflix.py`

Note this can take a while to run as it iterates over ~ 600MB of `combined_data_*.txt.gz` files and writes out a sqlite file.

The sqlite files is written to `./data/interim/netflix.db` and is roughly 9.6GB

3. Run `netflix_run.py` to demo use of pytorch with the sqlite wrapper.

`netflix_run.py` is the main execution path and uses a few helper files:

- `netflix_data.py` implements the torch Dataset interface, wrapping the sqlite database created in step 2.
- `netflix_models.py` implements a few basic NN-based recommender algorithms
- `netflix_tests.py` wraps instantiation of models and tests a small 1-batch fit.

## TODO

- Add requirements.txt
- Refactor implementation for general-purpose use

## Contributions

PRs welcome. 