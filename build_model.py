import os
import math
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle


class XGBoost_model:
    '''
        XGBoost for mapping PM2.5
    '''
    
    num_round = 2000
    
    def __init__(self, params, independent_v, dependent_v, training_data_path) -> None:
        self.params = params
        self.independent_v = independent_v
        self.dependent_v = dependent_v
        self.origin_dataset = pd.read_csv(training_data_path)    
        self.origin_dataset['doy'] = pd.to_datetime(self.origin_dataset['date']).dt.day_of_year
        self.nsamples = self.origin_dataset.shape[0]    
        num_te = int(0.2 * self.nsamples)
        num_val = int(0.1 * self.nsamples)
        num_tr = self.nsamples - num_val - num_te
        self.origin_dataset = shuffle(self.origin_dataset).reset_index(drop=True)
        self.training_data = self.origin_dataset[:num_tr]
        self.validation_data = self.origin_dataset[num_tr: num_tr + num_val]
        self.test_data = self.origin_dataset[num_tr + num_val:]
    
    @staticmethod
    def derive_data_label(dataset, independent_v, dependent_v):
        data = dataset.loc[:, independent_v]
        label = dataset.loc[:, dependent_v]
        return data, label
    
    def train(self):
        self.dtrain = xgb.DMatrix(self.derive_data_label(self.origin_dataset, self.independent_v, self.dependent_v)[0], 
                                  label=self.derive_data_label(self.origin_dataset, self.independent_v, self.dependent_v)[1])
        self.dvalid = xgb.DMatrix(self.derive_data_label(self.validation_data, self.independent_v, self.dependent_v)[0], 
                                  label=self.derive_data_label(self.validation_data, self.independent_v, self.dependent_v)[1])
        self.dtest =  xgb.DMatrix(self.derive_data_label(self.test_data, self.independent_v, self.dependent_v)[0], 
                                  label=self.derive_data_label(self.test_data, self.independent_v, self.dependent_v)[1])
        evallist = [(self.dtrain, 'train'), (self.dtest, 'eval')]
        bst = xgb.train(self.params, self.dtrain, self.num_round, evallist, early_stopping_rounds=10)
        return bst
    
    def ultimate_train(self):
        dtrain = xgb.DMatrix(self.derive_data_label(self.origin_dataset, self.independent_v, self.dependent_v)[0], 
                             label=self.derive_data_label(self.origin_dataset, self.independent_v, self.dependent_v)[1])
        bst = xgb.train(self.params, dtrain, self.num_round)
        self.bst = bst
        return bst

    def nfold_CV(self, nfold):
        pass
    
    def predict(self, input_dir, output_dir):
        file_list = os.listdir(input_dir)
        for filename in file_list:
            df = pd.read_csv(os.path.join(input_dir, filename))   
            if df.shape[0]==0:
                continue
            else:
                df['doy'] = pd.to_datetime(df['date']).dt.day_of_year 
                ypred = self.bst.predict(xgb.DMatrix(data=df.loc[:, self.independent_v]))  
                df_pre = pd.DataFrame({'row': df['row'], 'col': df['col'], 'date': df['date'], 'PM25': ypred})
                df_pre.to_csv(os.path.join(output_dir, filename))
            


    
    
    

    
    
    
    
    
