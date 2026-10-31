import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
def validate_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"file does not exist")
    if not os.path.isfile(path):
        raise ValueError(f"given path is not a file")
    if not path.lower().endswith("csv"):
        raise ValueError(f"the file is not csv")
    else:
        return True
def validate_dataset(x):
    if not isinstance(x,pd.DataFrame):
        raise TypeError('Given object is not a dataframe')
    if x.empty:
        raise ValueError('the dataframe does not have any rows')
    if x.columns.duplicated().any():
        raise ValueError('the dataframe has duplicated columns')
    return x.copy()

# checking the dataset characteristics
def dataset_info(df):
    df=validate_dataset(df)
    print('==================================')
    print('shape of the dataset :',df.shape)
    print('no.of rows           :',df.shape[0])
    print('no.of columns        :',df.shape[1])
    print('column names         :',df.columns.tolist())
    missing_values=df.isnull().mean()*100
    print('missing_values')
    print(missing_values)
    print('==================================')

#seperating datatypes
def seperate_dtypes(df,numeric=True,categorical=True):
    num_columns=df.select_dtypes(include=['int','float']).columns.tolist()
    cat_columns=df.select_dtypes(include=['object']).columns.tolist()
    if(numeric and categorical):
        return num_columns,cat_columns
    if(numeric):
        return num_columns
    if(categorical):
        return cat_columns

##seperating target from dataset
def seperate_target(df,target):
    if target not in df.columns:
        raise ValueError('target is not in dataframe')
    X=df.drop(columns=[target])
    y=df[target]
    return X,y  

### preprocessing pipeline
class RandomImputer(BaseEstimator,TransformerMixin):
    def __init__(self,random_state=42):
        self.random_state=random_state
    def fit(self,X_train,y_train=None):
        X=pd.DataFrame(X_train).copy()
        self.columns_=X.columns
        self.values_={}
        for col in self.columns_:
            self.values_[col]=X[col].dropna().unique()
        return self
    def transform(self,X_train,y_train=None):
        X=pd.DataFrame(X_train).copy()
        rng=np.random.default_rng(self.random_state)
        for col in self.columns_:
            mask=X[col].isna()
            n_missing=mask.sum()
            if n_missing>0:
                random_values=rng.choice(self.values_[col],size=n_missing,replace=True)
                X.loc[mask,col]=random_values
        return X

"""
- for age and categorical columns we will use random imputation
- for remaing columns use mean/median
- we have to onehot encode the categorical columns
- scale the numerical columns
"""

def create_preprocessor(df,age_column='age',target='addicted_label',imp_strategy='mean'):
    num_cols,cat_cols=seperate_dtypes(df)
    if age_column in num_cols:
        num_cols.remove(age_column)
    transformers=[]
    ##age_column pipeline
    if age_column in df.columns:
        age_pipeline=Pipeline([
           ("imputer",RandomImputer()),
            ("scaler",StandardScaler())
        ])
        transformers.append(('age_pipe',age_pipeline,[age_column]))
    if num_cols:
        num_pipeline=Pipeline([
            ('imputer',SimpleImputer(strategy=imp_strategy)),
            ('scaler',StandardScaler())
        ])
        transformers.append(('num_pipe',num_pipeline,num_cols))
    if cat_cols:
        cat_pipeline=Pipeline([
            ('imputer',RandomImputer()),
        ('ohe',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
        ])
        transformers.append(('categorical',cat_pipeline,cat_cols))
    transformer=ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    return transformer

    





    
