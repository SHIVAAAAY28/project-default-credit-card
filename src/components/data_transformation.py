import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd


from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OneHotEncoder # One hot encoding Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
                    
            
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_cols=["LIMIT_BAL","AGE","BILL_AMT1","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
            categor_cols=["SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2","PAY_4"]
            
            # Define the custom ranking for each ordinal variable
            sex_cat=['1', '2']
            education_cat=["0","1","2","3","4","5","6"]
            marriage_cat=["0","1","2","3"]
            pay0_cat=["-2","-1","0","1","2","3","4","5","6","7","8"]
            pay2_cat=["-2","-1","0","1","2","3","4","5","6","7","8"]
            pay4_cat=["-2","-1","0","1","2","3","4","5","6","7","8"]
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())

                ]

            )
            #Categorical onehote encoder pipeline
            cat_onehot_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ("onehotencoder",OneHotEncoder(categories=[sex_cat,education_cat,marriage_cat,pay0_cat,pay2_cat,pay4_cat],sparse=False)),
                ("scaler",StandardScaler())
                ]

            )



            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ("cat_onehot_pipeline",cat_onehot_pipeline,categor_cols)

            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
            
            train_df["SEX"]=train_df["SEX"].astype("str")
            train_df["EDUCATION"]=train_df["EDUCATION"].astype("str")
            train_df["MARRIAGE"]=train_df["MARRIAGE"].astype("str")
            train_df["PAY_0"]=train_df["PAY_0"].astype("str")
            train_df["PAY_2"]=train_df["PAY_2"].astype("str")
            train_df["PAY_4"]=train_df["PAY_4"].astype("str")
            
            test_df["SEX"]=test_df["SEX"].astype("str")
            test_df["EDUCATION"]=test_df["EDUCATION"].astype("str")
            test_df["MARRIAGE"]=test_df["MARRIAGE"].astype("str")
            test_df["PAY_0"]=test_df["PAY_0"].astype("str")
            test_df["PAY_2"]=test_df["PAY_2"].astype("str")
            test_df["PAY_4"]=test_df["PAY_4"].astype("str")



            target_column_name = 'default_payment_next_month'
            drop_columns = [target_column_name,"ID","ID_a"]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)