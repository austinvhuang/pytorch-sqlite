from math import ceil
from random import shuffle
import sqlite3
import torch.utils.data
import torch as T
from typing import Tuple, List, NamedTuple


def get_id_sets(netflix):
    customers = {
        cust
        for (cust,) in netflix.query(
            "select distinct customer_id from ratings order by customer_id"
        )
    }
    movies = {
        movie for (movie,) in netflix.query("select distinct movie_id from titles")
    }
    return (customers, movies)


class NetflixBatch(NamedTuple):
    items: T.LongTensor
    users: T.LongTensor
    ratings: T.Tensor


class NetflixUsers(T.utils.data.Dataset):
    def __init__(self, dbfile="data/interim/netflix.db", batch_size=1):
        self.conn = sqlite3.connect(dbfile)
        self.cursor = self.conn.cursor()
        customers_set, movies_set = get_id_sets(self)
        n_users = len(customers_set)
        n_items = len(movies_set)
        self.batch_size = batch_size
        self.item_id2idx = dict(zip(sorted(movies_set), range(n_items)))
        self.user_id2idx = dict(zip(sorted(customers_set), range(n_users)))
        self.user_idx2id = dict(zip(range(n_users), sorted(customers_set)))

    def n_users(self) -> int:
        """Number of users in the dataset"""
        return len(self.user_idx2id)

    def n_items(self) -> int:
        """Number of items in the dataset"""
        return len(self.item_id2idx)

    def __len__(self) -> int:
        """length special method"""
        # return self.n_users()
        return ceil(self.n_users() / self.batch_size)

    def get_single(self, index: int) -> Tuple[T.LongTensor, T.Tensor]:
        items = self.query(
            "select movie_id, customer_id, rating from ratings where customer_id = %d"
            % self.user_idx2id[index]
        )
        return self.process_query(items)

    def get_list(self, indices: list) -> Tuple[T.LongTensor, T.Tensor]:
        """__getitem__ handler for list inputs"""
        lst = [self.user_idx2id[index] for index in indices]
        items = self.query(
            "select movie_id, customer_id, rating from ratings where customer_id in (%s)"
            % ",".join([str(x) for x in lst])
        )
        return self.process_query(items)

    def get_slice(self, index: slice) -> Tuple[T.LongTensor, T.Tensor]:
        """__getitem__ handler for slice inputs"""
        (start, stop, step) = (
            index.start,
            index.stop,
            1 if index.step is None else index.step,
        )
        assert not start is None and not stop is None
        return self.get_list(list(range(start, stop, step)))

    def process_query(
        self, items: List[Tuple[int, int, int]]
    ) -> Tuple[T.LongTensor, T.LongTensor, T.Tensor]:
        """__getitem__ post-query processing"""
        result = {}
        for rating in range(1, 6):  # 1, 2, 3, 4, 5
            """item[0] : movie id, item[1] : customer_id, item[2] : rating
            (assumes query is of the form "select movie_id, rating, customer_id ..." )"""
            item_ids = [item[0] for item in items if item[2] == rating]
            user_ids = [item[1] for item in items if item[2] == rating]
            result[rating] = NetflixBatch(
                items=T.LongTensor([self.item_id2idx[item] for item in item_ids]),
                users=T.LongTensor([self.user_id2idx[user] for user in user_ids]),
                ratings=T.ones(len(item_ids)) * rating,
            )
        return NetflixBatch(
            items=T.cat([result[rating].items for rating in range(1, 6)]),
            users=T.cat([result[rating].users for rating in range(1, 6)]),
            ratings=T.cat([result[rating].ratings for rating in range(1, 6)]),
        )

    def __getitem__(self, index) -> Tuple[T.LongTensor, T.Tensor]:
        """get all movie id indexes for a given user index"""
        if isinstance(index, slice):
            return self.get_slice(index)
        if isinstance(index, list):
            return self.get_list(index)
        if isinstance(index, int):
            return self.get_single(index)
        raise ValueError("Type of %s not supported by __getitem()__" % str(index))

    def random_ordering(self) -> List[int]:
        lst = list(range(len(self)))
        shuffle(lst)
        return lst

    def query(self, query_string: str) -> List[Tuple]:
        """run a query and return the result"""
        self.cursor.execute(query_string)
        return self.cursor.fetchall()

    def __iter__(self):
        return NetflixIterator(self, batch_size=self.batch_size)


class NetflixIterator:
    def __init__(self, netflix, batch_size=1):
        self.ordering = list(range(netflix.n_users()))
        shuffle(self.ordering)
        self.index = 0
        self.iterable = netflix
        self.batch_size = batch_size

    def __next__(self):
        indexes = self.ordering[self.index : (self.index + self.batch_size)]
        result = self.iterable[indexes]
        self.index = self.index + self.batch_size
        return result
