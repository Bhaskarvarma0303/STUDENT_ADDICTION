import numpy as np
import os
import joblib 
import pandas as pd
def predict_addiction(input_list):
    model_path='artifacts/model.pkl'
    if not  os.path.exists(model_path):
        raise FileNotFoundError('model not found')
    if isinstance(input_list,pd.DataFrame):
        df_input=input_list
    # elif isinstance(input_list,np.ndarray):
    #     df_input=pd.DataFrame(input_list)
    elif isinstance(input_list,dict):
        df_input=pd.DataFrame([input_list])
    else:
        raise ValueError('the input must be a dictionary or dataframe')
    model=joblib.load(model_path)
    y_pred=model.predict(df_input)
    try:
        probablities=model.predict_proba(df_input)
        confidence=max(probablities[0])*100
    except AttributeError:
        confidence=0
    # print('='*30)
    # print('results')
    # print("student is addicted"if y_pred==1 else "student is not addicted")
    # print(f"confidence percentage: {confidence}")
    return y_pred

if __name__=='__main__':
    cols=['age','daily_screen_time_hours','social_media_hours','gaming_hours','work_study_hours','sleep_hours','notifications_per_day','app_opens_per_day','weekend_screen_time','gender','stress_level','academic_work_impact']
    values=[24.0,np.nan,1.83,1.59,2.11,7.46,122.0,38.0,8.63,'Male','Medium','No']
    df_input=pd.DataFrame([values],columns=cols)
    input={}
    for cols,values in zip(cols,values):
        input[cols]=values
    print(input)
    predict_addiction(input)
    
    