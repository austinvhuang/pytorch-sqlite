import codecs
import gzip
import time
import pandas as pd
import sqlite3
from tqdm import tqdm


def linecount(filename, encoding="utf-8"):
    """Count the number of lines in a file"""
    if filename.lower().endswith(".gz"):
        with gzip.open(filename, "rt", encoding=encoding) as contents:
            for i, _ in enumerate(contents):
                pass
            return i + 1
    else:
        with open(filename, "r", encoding=encoding) as contents:
            for i, _ in enumerate(contents):
                pass
            return i + 1


def int_or_none(string_val):
    """Return integer version of string or None if a cast fails"""
    try:
        return int(string_val)
    except:
        return None


def ingest_ratings(filename):
    """Transform the ratings table into a dataframe"""
    print(filename)
    movie_id = None
    df_dict = {"movie_id": [], "customer_id": [], "rating": [], "date": []}
    encoding = "ISO-8859-1"
    total = linecount(filename, encoding=encoding)
    with gzip.open(filename, "rt", encoding=encoding) as contents:
        for i, line in enumerate(tqdm(contents, total=total)):
            if line.endswith(":\n"):
                movie_id = int_or_none(line[:-2])
            else:
                df_dict["movie_id"].append(movie_id)
                (customer_id, rating, date) = line.split(",")
                df_dict["customer_id"].append(int_or_none(customer_id))
                df_dict["rating"].append(int_or_none(rating))
                # df_dict["date"].append(time.strptime(date[:-1], "%Y-%m-%d"))
                df_dict["date"].append(date[:-1])
    return pd.DataFrame(df_dict)


def ingest_titles(filename):
    """Transform the titles table into a dataframe"""
    df_dict = {"movie_id": [], "year": [], "title": []}
    encoding = "ISO-8859-1"
    reader = codecs.getreader(encoding)
    total = linecount(filename, encoding=encoding)
    with open(filename, "r", encoding=encoding) as contents:
        for i, line in enumerate(tqdm(contents, total=total)):
            split = line.split(",")
            movie_id = int(split[0])
            year = int_or_none(split[1])
            title = (",".join(split[2:]))[:-1]
            df_dict["movie_id"].append(movie_id)
            df_dict["year"].append(year)
            df_dict["title"].append(title)
    return pd.DataFrame(df_dict)


def add_identifier_column(df, prefix):
    """Add a unique identifier column to a dataframe"""
    df[prefix + "_id"] = range(len(df.index))
    return df


def create_indexes(conn):
    """Create column indexes"""
    cursor = conn.cursor()
    cursor.execute('create index "ix_titles_movie_id_index" on titles (movie_id);')
    cursor.execute('create index "ix_titles_year_index" on titles (year);')
    cursor.execute('create index "ix_titles_id_index" on titles (titles_id);')

    cursor.execute('create index "ix_ratings_movie_id_index" on ratings (movie_id);')
    cursor.execute('create index "ix_customer_id_index" on ratings (customer_id);')
    cursor.execute('create index "ix_ratings_id_index" on ratings (ratings_id);')
    cursor.execute('create index "ix_rating_index" on ratings (rating);')
    cursor.close()


if __name__ == "__main__":
    print("Loading raw files")
    df = ingest_titles("data/raw/netflix/movie_titles.csv")
    df = add_identifier_column(df, "titles")
    conn = sqlite3.connect("data/interim/netflix.db")
    df.to_sql("titles", conn, if_exists="replace")

    print("Loading ratings files")
    file_list = [
        ("data/raw/netflix/combined_data_%d.txt.gz" % idx) for idx in range(1, 5)
    ]
    df_list = [ingest_ratings(f) for f in file_list]

    print("Concatenating data frames")
    df = pd.concat(df_list)
    df = add_identifier_column(df, "ratings")

    print("Writing to sqlite")
    conn = sqlite3.connect("data/interim/netflix.db")
    df.to_sql("ratings", conn, if_exists="replace")

    print("Creating indexes")
    create_indexes(conn)

    conn.close()

    print("Done")
