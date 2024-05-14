import pandas as pd
import numpy as np
import os
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts","raw.csv")
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            data = pd.read_csv(Path(os.path.join("notebooks/data","gemstone.csv")))
            logging.info("Data read as dataframe")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved in artifacts")

            logging.info("Train test split")
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Train test split completed")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.train_data_path)), exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            logging.info("Train data saved in artifacts")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.test_data_path)), exist_ok=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Test data saved in artifacts")

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info("Error occurred in data ingestion")
            raise CustomException(e, sys)

        else:
            logging.info("Data ingestion completed successfully")