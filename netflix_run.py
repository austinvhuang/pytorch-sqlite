import pandas as pd
import sqlite3
import torch as T
from pdb import set_trace

from typing import Tuple, List, NamedTuple

from netflix_data import NetflixBatch, NetflixUsers
from netflix_tests import mf_1batch_test, densenet_1batch_test, wad_1batch_test
from netflix_models import (
    run_mf,
    run_dense,
    ModelConfig,
    MatrixFactorization,
    DenseNet,
    WideAndDeep,
)


if __name__ == "__main__":

    print("Loading data")
    netflix = NetflixUsers(batch_size=100)

    print("Running training")

    # mf = run_mf(netflix)
    print("  Matrix Factorization")
    mf_1batch_test(netflix, show_interval=5, n_steps=20)

    # dense = run_dense(netflix)
    print("  Dense Network")
    densenet_1batch_test(netflix, show_interval=5, n_steps=20)

    print("  Wide and Deep Network")
    wad_1batch_test(netflix, show_interval=5, n_steps=20)

    print("Done")
