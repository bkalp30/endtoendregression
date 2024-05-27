from src.DimondPricePrediction.components.data_ingestion import DataIngestion
import os
import sys
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException
import pandas as pd

from src.DimondPricePrediction.components.data_transformation import DataTransformation
from src.DimondPricePrediction.components.model_trainer import ModelTrainer
from src.DimondPricePrediction.components.model_evaluation import ModelEvaluation


data_ingestion = DataIngestion()
train_path, test_path = data_ingestion.initiate_data_ingestion()

data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initiate_data_transformation(train_path=train_path, test_path=test_path)

model_trainer = ModelTrainer()
model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr)

model_evaluation = ModelEvaluation()
model_evaluation.initiate_model_evaluation(train_array=train_arr, test_array=test_arr)
