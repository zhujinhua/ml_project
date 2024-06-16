import time
import dataLoader


def main():
    start_time = time.time()
    X, y = dataLoader.load_horse_rental_data()
    print(time.time() - start_time, "s")

    print("columns:", len(X[0]))
    print("X raws ：", len(X))
    print("y raws: ", len(y))


if __name__ == "__main__":
    main()