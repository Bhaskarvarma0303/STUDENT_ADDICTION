import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.predict import predict_addiction
st.set_page_config(layout='wide',page_title='student_Addiction_platform',)
st.title(':green[STUDENT ADDICTION PLATFORM]',text_alignment='center')
st.markdown('<b> A platform where it predicts the student addiction',unsafe_allow_html=True,text_alignment='center')
st.subheader(':green[Please Enter These Values]')
with st.form("prediction_form"):
        st.subheader("Student Profile Inputs")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=30, value=19)
            screen_time = st.number_input("Daily Screen Time (Hours)", min_value=0.0, max_value=24.0, value=np.nan,placeholder='unknown')
            study_hours = st.number_input("Daily Study Time (Hours)", min_value=0.0, max_value=24.0, value=np.nan,placeholder='unknown')
            social_media_hours = st.number_input("Social Media Time (Hours)", min_value=0.0, max_value=24.0, value=np.nan,placeholder='unknown')
            gaming_hours=st.number_input("Gaming Time (Hours)", min_value=0.0, max_value=24.0, value=np.nan,placeholder='unknown')
            work_study_hours=st.number_input("Work Study (Hours)", min_value=0.0, max_value=24.0, value=np.nan,placeholder='unknown')
        with col2:
            notifications=st.number_input("notifications per day",min_value=0,max_value=1000,value=300)
            app_opens=st.number_input("App Opens Per Day ",min_value=0,max_value=500,value=100)
            sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=np.nan,placeholder='unknown')
            weekend_screen_time=st.number_input("Weekend Screen Time (Hours)",min_value=0.0,max_value=15.0,value=np.nan,placeholder='unknown',)
            gender = st.selectbox("Gender", ["None","Male", "Female","Other"])
            stress_level=st.selectbox('Stress Level',["None","Low","Medium","High"])
        academic_impact=st.selectbox('Academic Work Impact',['None','Yes','No'],placeholder='Unknown')
        if academic_impact is "None":
             academic_impact=np.nan #or dicerctly convert while creating a dataframe
            # Add more inputs here as needed matching your dataset columns!
        submit=st.form_submit_button("predict the addiction")
input={
     'age':age,
      'daily_screen_time_hours':screen_time,
      'social_media_hours':social_media_hours,
      'gaming_hours':gaming_hours,
      'work_study_hours':work_study_hours,
      'sleep_hours':study_hours,
      'notifications_per_day':notifications,
      'app_opens_per_day':app_opens,
      'weekend_screen_time':weekend_screen_time,
      'gender':np.nan if gender is  'None' else gender,
      'stress_level':np.nan if stress_level is  'None' else stress_level
      ,'academic_work_impact':academic_impact
      }
prediction=predict_addiction(input)
if prediction==1:
     st.write("student is addicted")
else:
     st.write("student is not addicted")
