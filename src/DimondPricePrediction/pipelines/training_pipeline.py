from src.DimondPricePrediction.components.data_ingestion import DataIngestion
import os
import sys
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException
import pandas as pd

data_ingestion = DataIngestion()
data_ingestion.initiate_data_ingestion()