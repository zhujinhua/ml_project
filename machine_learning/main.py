import time

import pandas as pd

import dataLoader
from machine_learning.train_model import train_model


def main():
    start_time = time.time()
    X, y = dataLoader.load_horse_rental_data()
    print(time.time() - start_time, "s")

    print("columns:", len(X[0]))
    print("X raws ：", len(X))
    print("y raws: ", len(y))

    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)

    train_model(df_X, df_y)


if __name__ == "__main__":
    main()
