import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def load_data(path):
    return pd.read_csv(path)

def build_matrix(df_train, df_val, df_test):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient='records'))
    X_val = dv.transform(df_val.to_dict(orient='records'))
    X_test = dv.transform(df_test.to_dict(orient='records'))
    return dv, X_train, X_val, X_test

def main(data_path, target, out_dir, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(data_path)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
    df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=random_state)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)


    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    del df_train[target]
    del df_val[target]
    del df_test[target]

    dv, X_train, X_val, X_test = build_matrix(df_train, df_val, df_test)

    # candidate models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state)
    }

    results = {}
    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        results[name] = {"model": model, "rmse": rmse}
        print(f"  RMSE (val): {rmse:.4f}")

    # select best
    best_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    best_model = results[best_name]["model"]
    best_rmse = results[best_name]["rmse"]
    print(f"Best model: {best_name} with RMSE {best_rmse:.4f}")

    # final evaluation on test
    y_test_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Test RMSE of best model: {test_rmse:.4f}")

    # save artifacts
    model_path = os.path.join(out_dir, "model.pkl")
    dv_path = os.path.join(out_dir, "dv.pkl")
    with open(model_path, "wb") as f_out:
        pickle.dump(best_model, f_out)
    with open(dv_path, "wb") as f_out:
        pickle.dump(dv, f_out)
    print("Saved model to", model_path)
    print("Saved dict vectorizer to", dv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="../data/shopping_behavior_updated.csv", help="CSV path")
    parser.add_argument("--target", default="review_rating", help="target column name")
    parser.add_argument("--out-dir", default="../model", help="output dir for model artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args.data_path, args.target, args.out_dir, args.random_state)
