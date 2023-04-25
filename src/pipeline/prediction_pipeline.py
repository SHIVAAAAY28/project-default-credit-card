import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 LIMIT_BAL:int,
                 AGE:int,
                 BILL_AMT1:int,
                 PAY_AMT1:int,
                 PAY_AMT2:int,
                 PAY_AMT3:int,
                 PAY_AMT4:int,
                 PAY_AMT5:int,
                 PAY_AMT6:int,
                 SEX:str,
                 EDUCATION:str,
                 MARRIAGE:str,
                 PAY_0:str,
                 PAY_2:str,
                 PAY_4:str):
        
        self.LIMIT_BAL=LIMIT_BAL
        self.AGE=AGE
        self.BILL_AMT1=BILL_AMT1
        self.PAY_AMT1=PAY_AMT1
        self.PAY_AMT2=PAY_AMT2
        self.PAY_AMT3=PAY_AMT3
        self.PAY_AMT4 = PAY_AMT4
        self.PAY_AMT5 = PAY_AMT5
        self.PAY_AMT6 = PAY_AMT6
        self.SEX = SEX
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.PAY_0 = PAY_0
        self.PAY_2 = PAY_2
        self.PAY_4 = PAY_4
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict ={
                'LIMIT_BAL':[self.LIMIT_BAL],
                'AGE':[self.AGE],
                'BILL_AMT1':[self.BILL_AMT1],
                'PAY_AMT1':[self.PAY_AMT1],
                'PAY_AMT2':[self.PAY_AMT2],
                'PAY_AMT3':[self.PAY_AMT3],
                'PAY_AMT4':[self.PAY_AMT4],
                'PAY_AMT5':[self.PAY_AMT5],
                'PAY_AMT6':[self.PAY_AMT6],
                'SEX':[self.SEX],
                'EDUCATION':[self.EDUCATION],
                'MARRIAGE':[self.MARRIAGE],
                'PAY_0':[self.PAY_0],
                'PAY_2':[self.PAY_2],
                'PAY_4':[self.PAY_4]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
