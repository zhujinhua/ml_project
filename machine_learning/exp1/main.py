import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import dataLoader


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    mae = round(mean_absolute_error(y_true=y_test, y_pred=rf_pred), 3)
    print(mae)


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
