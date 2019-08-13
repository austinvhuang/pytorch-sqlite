# Example using Sqlite with a Pytorch Dataset Interface

This repo demonstrates using sqlite used as a backend for pytorch's `Torch.utils.data.Dataset` interface.

This code uses the netflix prize dataset as an example, but the same idea can be generally applied.

## Running

### 1. Download netflix prize data

The data should be downloaded from:

https://www.kaggle.com/netflix-inc/netflix-prize-data

and extracted into `./data/raw/netflix/`. To keep file sizes manageable, gzip all .txt files. The following scripts assume files are gzipped.

Thus, the contents of `./data/raw/netflix/` should be:

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

### 2. Run `ingest_netflix.py` to construct a sqlite db

Note this can take a while to run as it iterates over ~ 600MB of `combined_data_*.txt.gz` files and writes out a sqlite file. Writing the sqlite file in particular takes a while.

The sqlite files is written to `./data/interim/netflix.db` and is roughly 9.6GB.

### 3. Run `netflix_run.py` to demo use of pytorch with the sqlite wrapper.

`netflix_run.py` is the main execution path and uses a few helper files:

- `netflix_data.py` implements the torch Dataset interface, wrapping the sqlite database created in step 2.
- `netflix_models.py` implements a few basic NN-based recommender algorithms
- `netflix_tests.py` wraps instantiation of models and tests a small 1-batch fit.

## TODO

- Add requirements.txt
- Refactor implementation for general-purpose use

## Contributions

PRs welcome. 