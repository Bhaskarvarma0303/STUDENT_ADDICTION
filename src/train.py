import pandas as pd
import joblib,os
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix
from src.preprocessing import validate_path,dataset_info,seperate_dtypes,seperate_target,create_preprocessor
def main():
    dataset_path='Datasets/Student_Addiction.csv'
    output_path='artifacts/model.pkl'
    if os.path.exists(output_path):
        os.remove(output_path)
        print('old model successfully deleted')
    validate_path(dataset_path)
    df=pd.read_csv(dataset_path)
    print("dataset loaded successfully")
    ##seperating target
    X,y=seperate_target(df,target='addicted_label')
    ##dropping unnecessary_columns
    X.drop(columns=['id'],inplace=True)
    ##train_test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
    print(f"X_train shape: {X_train.shape}")
    print(f'X_train type{type(X_train)}')
    print(f"no.of rows to be trained {X_train.shape[0]}")
    ##training the model
    preprocessor=create_preprocessor(X_train,scaling=False)
    classifier=xgboost.XGBClassifier()
    model=Pipeline([
        ("preprocessor",preprocessor),
        ("classifier",classifier)
    ])
    print("started model training....")
    model.fit(X_train,y_train)
    print("training successfully completed")
    ##prediction 
    print(f'test_data type:{type(X_test)}')
    y_pred=model.predict(X_test)
    print("prediction completed")
    print("results")
    print(f"accuracy_score:{accuracy_score(y_test,y_pred)}")
    print('confusion matrix')
    print(confusion_matrix(y_test,y_pred))
    ##model saving
    joblib.dump(model,output_path)
    print(f"model successfully saved to {output_path}")
if __name__=='__main__':
    main()


        