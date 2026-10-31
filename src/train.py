import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import validate_path,dataset_info,seperate_dtypes,seperate_target,create_preprocessor
dataset_path='Datasets/Student_Addiction.csv'
if validate_path(dataset_path):
    df=pd.read_csv(dataset_path)
X,y=seperate_target(df,target='addicted_label')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
transformer=create_preprocessor(X_train)
updated_df=transformer.fit_transform(X_train)
