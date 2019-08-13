import pandas as pd
import sqlite3
from tensorboardX import SummaryWriter
import torch as T

from netflix_models import MatrixFactorization, DenseNet, WideAndDeep


def test_netflix() -> None:
    """Check that queries to netflix sqlite file work"""
    conn = sqlite3.connect("data/interim/netflix.db")
    print("\nRatings\n====================")
    query = "select * from ratings limit 5;"
    print(pd.read_sql_query(query, conn))
    print("\nTitles\n====================")
    query = "select * from titles limit 5;"
    print(pd.read_sql_query(query, conn))
    conn.close()


"""
1-batch overfit tests
"""


def test_1batch(netflix, model, optimizer, show_interval, n_steps):
    loss = T.nn.MSELoss()
    writer = SummaryWriter()
    (batch_items, batch_users, batch_ratings) = next(iter(netflix))
    for step in range(n_steps):
        prediction = model(batch_users, batch_items)
        mse = loss(prediction, batch_ratings)
        mse.backward()
        optimizer.step()
        writer.add_scalar("data/overfit_1batch", mse, step)
        print("Iter %d / Loss : %s" % (step, mse))
    writer.export_scalars_to_json("./data/interim/overfit_1batch.json")


def mf_1batch_test(netflix, show_interval, n_steps):
    """matrix factorization overfit to 1 batch"""
    model = MatrixFactorization(netflix.n_users(), netflix.n_items(), n_factors=100)
    optimizer = T.optim.SparseAdam(model.parameters(), lr=1e-3)
    test_1batch(netflix, model, optimizer, show_interval, n_steps)


def densenet_1batch_test(netflix, show_interval, n_steps):
    """dense net overfit to 1 batch"""
    model = DenseNet(
        netflix.n_users(), netflix.n_items(), user_factors=50, item_factors=50
    )
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)
    test_1batch(netflix, model, optimizer, show_interval, n_steps)


def wad_1batch_test(netflix, show_interval, n_steps):
    """matrix factorization overfit to 1 batch"""
    model = WideAndDeep(
        netflix.n_users(), netflix.n_items(), embed_dim=50, hidden_dim=50
    )
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)
    test_1batch(netflix, model, optimizer, show_interval, n_steps)
