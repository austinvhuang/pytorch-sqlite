from tensorboardX import SummaryWriter
import torch as T
import torch.utils.data
from tqdm import tqdm
from pdb import set_trace

from typing import Tuple, List, NamedTuple


class ModelConfig(NamedTuple):
    model: T.nn.Module
    optimizer: T.optim.Optimizer


class MatrixFactorization(T.nn.Module):
    def __init__(self, user_dim, item_dim, n_factors):
        super().__init__()
        self.users = T.nn.Embedding(user_dim, n_factors, sparse=True)
        self.items = T.nn.Embedding(item_dim, n_factors, sparse=True)

    def forward(self, user, item):
        # note this is vectorized so that item can be a tensor
        return (self.users(user) * self.items(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)

    def embed_user(self, user):
        return self.users(user)

    def embed_item(self, item):
        return self.items(item)


class DenseNet(T.nn.Module):
    def __init__(self, user_dim, item_dim, user_factors, item_factors):
        super().__init__()
        hidden_factors = 100
        self.user_factors = torch.nn.Embedding(user_dim, user_factors)
        self.item_factors = torch.nn.Embedding(item_dim, item_factors)
        self.hidden = torch.nn.Linear(user_factors + item_factors, hidden_factors)
        self.output = torch.nn.Linear(hidden_factors, 1)

    def forward(self, users, items):
        users_embed = self.user_factors(users)
        items_embed = self.item_factors(items)
        h1 = T.cat([users_embed, items_embed], 1)
        h1_relu = T.relu(self.hidden(h1))
        return self.output(h1_relu)

    def predict(self, users, items):
        output_scores = self.forward(users, items)
        return output_scores


class WideAndDeep(T.nn.Module):
    def __init__(self, user_dim, item_dim, embed_dim, hidden_dim):
        super().__init__()
        super(WideAndDeep, self).__init__()
        self.wide = T.nn.Linear(item_dim + user_dim, 1)
        self.embed_users = T.nn.Embedding(user_dim, embed_dim)
        self.embed_items = T.nn.Embedding(item_dim, embed_dim)
        self.hidden = T.nn.Linear(embed_dim * 2, hidden_dim)
        self.output = T.nn.Linear(hidden_dim, 1)

    def forward(self, users, items):
        users_embed = self.embed_users(users)
        items_embed = self.embed_items(items)
        merge = T.cat([users_embed, items_embed], 1)
        hidden = self.hidden(merge)
        return self.output(T.relu(hidden))

    def predict(self, users, items):
        return self.forward(users, items)


def run_model(netflix, model, optimizer, show_interval=5, n_epoch=2):
    loss = torch.nn.MSELoss()
    writer = SummaryWriter()
    step = 0
    for epoch in range(n_epoch):
        print("Epoch : %d" % epoch)
        for idx, (batch_items, batch_users, batch_ratings) in enumerate(tqdm(netflix)):
            prediction = model(batch_users, batch_items)
            mse = loss(prediction, batch_ratings)
            if T.isnan(mse):
                print("nan encountered")
                set_trace()
            mse.backward()
            optimizer.step()
            step += 1
            # writer.add_scalar('data/model', mse, step)
            if idx % show_interval == 0:
                print("Loss : %s" % mse)
    writer.export_scalars_to_json("./data/interim/all_scalars.json")
    writer.close()
    return model


def run_configuration(netflix, config, show_interval=5, n_epochs=2):
    model = config.model
    optimizer = config.optimizer
    run_model(netflix, model, optimizer, show_interval, n_epoch)


def run_mf(netflix, show_interval=5, n_epoch=2):
    model = MatrixFactorization(netflix.n_users(), netflix.n_items(), n_factors=100)
    optimizer = torch.optim.SparseAdam(model.parameters())
    return run_model(netflix, model, optimizer, show_interval, n_epoch)


def run_dense(netflix, show_interval=5, n_epoch=2):
    model = DenseNet(
        netflix.n_users(), netflix.n_items(), user_factors=50, item_factors=50
    )
    optimizer = torch.optim.Adam(model.parameters())
    return run_model(netflix, model, optimizer, show_interval, n_epoch)


def run_wad(netflix):
    model = WideAndDeep(
        netflix.n_users(), netflix.n_items(), embed_dim=50, hidden_dim=50
    )
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)
    return ModelConfig(model, optimizer)


## Configuration factories


def make_wad(netflix):
    model = WideAndDeep(
        netflix.n_users(), netflix.n_items(), embed_dim=50, hidden_dim=50
    )
    optimizer = T.optim.Adam(model.parameters(), lr=1e-3)
    return ModelConfig(model, optimizer)


def make_mf(netflix):
    model = MatrixFactorization(netflix.n_users(), netflix.n_items(), n_factors=100)
    optimizer = torch.optim.SparseAdam(model.parameters())
    return ModelConfig(model, optimizer)


def make_dense(netflix):
    model = DenseNet(
        netflix.n_users(), netflix.n_items(), user_factors=50, item_factors=50
    )
    optimizer = torch.optim.Adam(model.parameters())
    return ModelConfig(model, optimizer)
