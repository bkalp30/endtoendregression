
import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.DimondPricePrediction.exception import CustomException
from src.DimondPricePrediction.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.DimondPricePrediction.utils.utils import save_object   

@dataclass
class DataTransaformationConfig:
    preprocessor_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.transformation_config = DataTransaformationConfig()
        
    def get_data_transformation(self):
        try:
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline initiated")

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            return preprocessor
        
        except Exception as e:
            logging.info("Error occurred in data transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Read train test data complete")
            logging.info(f'Train data: {train_data.head()}') 
            logging.info(f'Train data: {test_data.head()}')

            preprocessor_obj = self.get_data_transformation()

            target_col_name = 'price'
            drop_cols = [target_col_name, 'id']

            # segregate the data with respect to dependent and independent feat
            input_train_df = train_data.drop(columns=target_col_name, axis=1)
            target_feature_train_df = train_data[target_col_name]

            input_test_df = test_data.drop(columns=drop_cols, axis=1)
            target_feature_test_df = test_data[target_col_name]

            input_train_df_arr = preprocessor_obj.fit_transform(input_train_df)
            input_test_df_arr = preprocessor_obj.transform(input_test_df)

            logging.info("Applying preprocessor object on train and test datasets")

            save_object(
                file_path = self.transformation_config.preprocessor_file_path,
                obj = preprocessor_obj
            )
            
            train_arr = np.c_[input_train_df_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_test_df_arr, np.array(target_feature_test_df)]

            return (
                train_arr,
                test_arr
            )
            

        except Exception as e:
            logging.info("Error occurred in data transformation")
            raise CustomException(e, sys)