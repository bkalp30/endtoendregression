import os
import mlflow
import mlflow.sklearn
import sys
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import numpy as np
from src.DimondPricePrediction.utils.utils import load_object
from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.logger import logging
from urllib.parse import urlparse

class ModelEvaluation:
    def __init__(self) -> None:
        pass

    def evl_metrics(self, actual, predicted):
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        return r2, rmse, mae

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            x_test, y_test = (test_array[:,:-1], test_array[:,-1])
            modelpath = os.path.join("artifacts","model.pkl")
            model = load_object(modelpath)

            mlflow.set_registry_uri("https://dagshub.com/bkalp30/endtoendregression.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
            with mlflow.start_run():
                predicted =  model.predict(x_test)
                (rmse, mae, r2) = self.evl_metrics(y_test, predicted)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
        except Exception as e:
            logging.info("Error occurred in initiate_model_trainer")
            raise CustomException(e, sys)