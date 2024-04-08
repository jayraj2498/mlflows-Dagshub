import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import mlflow
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
# Rest of your code follows...


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, LinearRegression


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)



    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)   
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]



    # Define parameters for each algorithm
    elasticnet_alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    elasticnet_l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    rf_n_estimators = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    rf_max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    gb_n_estimators = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    gb_learning_rate = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1

    svm_kernel = sys.argv[7] if len(sys.argv) > 7 else 'rbf'
    svm_C = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0

    lr_normalize = bool(sys.argv[9]) if len(sys.argv) > 9 else False

    # ElasticNet
    with mlflow.start_run():
        lr = ElasticNet(alpha=elasticnet_alpha, l1_ratio=elasticnet_l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(elasticnet_alpha, elasticnet_l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("elasticnet_alpha", elasticnet_alpha)
        mlflow.log_param("elasticnet_l1_ratio", elasticnet_l1_ratio)
        mlflow.log_metric("elasticnet_rmse", rmse)
        mlflow.log_metric("elasticnet_r2", r2)
        mlflow.log_metric("elasticnet_mae", mae)

    # Random Forest
    with mlflow.start_run():
        rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
        rf.fit(train_x, train_y)

        predicted_qualities_rf = rf.predict(test_x)

        (rmse_rf, mae_rf, r2_rf) = eval_metrics(test_y, predicted_qualities_rf)

        print("Random Forest model (n_estimators={}, max_depth={}):".format(rf_n_estimators, rf_max_depth))
        print("  RMSE: %s" % rmse_rf)
        print("  MAE: %s" % mae_rf)
        print("  R2: %s" % r2_rf)

        mlflow.log_param("rf_n_estimators", rf_n_estimators)
        mlflow.log_param("rf_max_depth", rf_max_depth)
        mlflow.log_metric("rf_rmse", rmse_rf)
        mlflow.log_metric("rf_r2", r2_rf)
        mlflow.log_metric("rf_mae", mae_rf)

    # Gradient Boosting
    with mlflow.start_run():
        gb = GradientBoostingRegressor(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate, random_state=42)
        gb.fit(train_x, train_y)

        predicted_qualities_gb = gb.predict(test_x)

        (rmse_gb, mae_gb, r2_gb) = eval_metrics(test_y, predicted_qualities_gb)

        print("Gradient Boosting model (n_estimators={}, learning_rate={}):".format(gb_n_estimators, gb_learning_rate))
        print("  RMSE: %s" % rmse_gb)
        print("  MAE: %s" % mae_gb)
        print("  R2: %s" % r2_gb)

        mlflow.log_param("gb_n_estimators", gb_n_estimators)
        mlflow.log_param("gb_learning_rate", gb_learning_rate)
        mlflow.log_metric("gb_rmse", rmse_gb)
        mlflow.log_metric("gb_r2", r2_gb)
        mlflow.log_metric("gb_mae", mae_gb)

    # SVM
    with mlflow.start_run():
        svm = SVR(kernel=svm_kernel, C=svm_C)
        svm.fit(train_x, train_y.values.ravel())  # SVR expects 1d array for y

        predicted_qualities_svm = svm.predict(test_x)

        (rmse_svm, mae_svm, r2_svm) = eval_metrics(test_y, predicted_qualities_svm)

        print("Support Vector Machine model (kernel={}, C={}):".format(svm_kernel, svm_C))
        print("  RMSE: %s" % rmse_svm)
        print("  MAE: %s" % mae_svm)
        print("  R2: %s" % r2_svm)

        mlflow.log_param("svm_kernel", svm_kernel)
        mlflow.log_param("svm_C", svm_C)
        mlflow.log_metric("svm_rmse", rmse_svm)
        mlflow.log_metric("svm_r2", r2_svm)
        mlflow.log_metric("svm_mae", mae_svm)

    # Linear Regression
    with mlflow.start_run():
        lr_model = LinearRegression(normalize=lr_normalize)
        lr_model.fit(train_x, train_y)

        predicted_qualities_lr = lr_model.predict(test_x)

        (rmse_lr, mae_lr, r2_lr) = eval_metrics(test_y, predicted_qualities_lr)

        print("Linear Regression model (normalize={}):".format(lr_normalize))
        print("  RMSE: %s" % rmse_lr)
        print("  MAE: %s" % mae_lr)
        print("  R2: %s" % r2_lr)

        mlflow.log_param("lr_normalize", lr_normalize)
        mlflow.log_metric("lr_rmse", rmse_lr)
        mlflow.log_metric("lr_r2", r2_lr)
        mlflow.log_metric("lr_mae", mae_lr)
