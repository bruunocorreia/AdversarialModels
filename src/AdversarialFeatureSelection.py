
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import pyspark

class AdversarialModel:
    
    def __init__(self, auc_threshold=0.5, num_feature_delete_step=1):
        self.auc_threshold = auc_threshold
        self.num_feature_delete_step = num_feature_delete_step
        self.features_to_delete = []
        
    def fit(self, x, y):
        
        if self.auc_threshold < 0.5 or self.auc_threshold > 1:
            raise ValueError('The auc_threshold parameter must be between 0.5 and 1.')
        
        if self.num_feature_delete_step < 1 or self.num_feature_delete_step >= x.shape[1]:
            raise ValueError('The num_feature_delete_step parameter must be greater than 1 and less than the number of features in x_train.')
        
        if isinstance(x, pd.DataFrame):
            pass
        elif isinstance(x, spark.sql.DataFrame):
            x = x.toPandas()
            x = x.toPandas()
        else:
            raise ValueError('x must be both pandas or spark dataframes.')

        if isinstance(y, pd.DataFrame):
            pass
        elif isinstance(y, pyspark.sql.Column):
            y = y.toPandas()
            y = y.toPandas()
        else:
            raise ValueError('y must be both pandas or both spark columns.')

        print("Initial number of features: ", len(x.columns))
        
        #Train and test split
        BrunoSeed = 18051996
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=BrunoSeed,stratify=y)
        # Start model
        adversarial_model = RandomForestClassifier(n_estimators=50,
                                                   max_depth=8,
                                                   random_state=18051996)
        adversarial_model.fit(x_train, y_train)

        # score train and test data
        proba_train = adversarial_model.predict_proba(x_train)[:, 1]
        proba_test = adversarial_model.predict_proba(x_test)[:, 1]

        # compute AUC
        auc_train = roc_auc_score(y_train, proba_train)
        auc_test = roc_auc_score(y_test, proba_test)

        print(f'First AUC on test: {auc_test:.4f}')
               
        while auc_test > self.auc_threshold:

            print('Drift detected')
            feature_importance_rf = (pd.DataFrame(adversarial_model.feature_importances_, index=x_train.columns)
                                     .sort_values(by=0, ascending=False))

            for delete_feature in range(0,self.num_feature_delete_step):
                add = feature_importance_rf.reset_index()['index'][delete_feature]
                self.features_to_delete.extend([add])

            print('Deleting features:')
            print(*self.features_to_delete, sep='\n')
            print('---------------------')

            x_train = x_train.drop(columns=self.features_to_delete,errors='ignore')
            x_test = x_test.drop(columns=self.features_to_delete,errors='ignore')

            # redefine new model
            adversarial_model = RandomForestClassifier(n_estimators=50, max_depth=8,random_state = 18051996)
            adversarial_model.fit(x_train, y_train)

            # score train and test data
            proba_train = adversarial_model.predict_proba(x_train)[:, 1]
            proba_test = adversarial_model.predict_proba(x_test)[:, 1]

            # compute AUC
            auc_train = roc_auc_score(y_train, proba_train)
            auc_test = roc_auc_score(y_test, proba_test)

            print(f'AUC on test: {auc_test:.4f}')

        print('Features to delete:')
        print(*self.features_to_delete, sep='\n')
        print(f"Optimal number of features: {len(x_train.columns)}")

        return self.features_to_delete
    
    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            print(self.features_to_delete)
            return data.drop(columns=self.features_to_delete)
        elif 'pyspark' in str(type(data)):
            columns_to_drop = [col for col in self.features_to_delete]
            return data.drop(*columns_to_drop)
        else:
            raise TypeError('Data must be a Pandas DataFrame or a PySpark DataFrame.')