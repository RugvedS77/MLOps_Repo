import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--max_depth", type=int, default=2)
    args = parser.parse_args()

    # Load data
    data = pd.read_csv("data/winequality-red.csv", sep=';')
    X = data.drop(["quality"], axis=1)
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # Train RandomForest
        rf = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_rmse = eval_metrics(y_test, rf_pred)
        mlflow.log_metric("rf_rmse", rf_rmse)

        # Log model
        mlflow.sklearn.log_model(rf, "random_forest_model")

        # Train XGBoost
        xgbr = xgb.XGBRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
        xgbr.fit(X_train, y_train)
        xgb_pred = xgbr.predict(X_test)
        xgb_rmse = eval_metrics(y_test, xgb_pred)
        mlflow.log_metric("xgb_rmse", xgb_rmse)

        mlflow.xgboost.log_model(xgbr, "xgboost_model")

        # Register best model
        if xgb_rmse < rf_rmse:
            result = mlflow.register_model(
                "runs:/{}/xgboost_model".format(run.info.run_id),
                "WineQualityModel"
            )
        else:
            result = mlflow.register_model(
                "runs:/{}/random_forest_model".format(run.info.run_id),
                "WineQualityModel"
            )

        print("Model Registered:", result.name, result.version)
