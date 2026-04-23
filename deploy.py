import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title = 'Sudent PASS / FAIL Predictor' , page_icon= '🎓')
st.title('🎓 Student Performance Predictor')
st.write("Enter a student's lifestyle and studying metrics below to predict their exam outcomes. ")

@st._cache_resource
def load_model():
    return joblib.load('model.pkl')

deployed_model = load_model()

st.subheader('Student Data')
col1,col2 = st.columns(2)
with col1:
    nam = st.text_input('Name of STUDENT')
    gen = st.selectbox('Gender of STUDENT' , ['Male', 'Female'])
    age = st.slider('Age of STUDENT',step=1,min_value=10 , max_value=25)
    parent_edu = st.slider('Parents Education Level' , step = 1 , min_value=1,max_value=9)
    family_income = st.slider('Family Income',step=500,min_value=10000,max_value=500000)
    study_time = st.slider('Study Time in Hours',step=0.1,min_value=0.0,max_value=24.0)

with col2:
    attendance = st.slider('Attendance Rate (%))' ,min_value=0,max_value=100,step=1)
    sleep_time = st.slider('Sleep Time in Hours' , step=0.1,min_value=0.0,max_value=14.0)
    stress = st.slider('Stress Level of STUDENT', step=1,min_value=1,max_value=10)
    motivation = st.slider('Motivation Level of STUDENT',step=1,min_value=0,max_value=100)
    private_tutor = st.selectbox('Does the STUDENT opts PRIVATE TUTORING',[True , False])
    internet_quality = st.slider("STUDENT's Internet Quality",step=1,min_value = 1 , max_value=5)

if st.button('Predict Outcome',type='primary'):
    new_student =  pd.DataFrame({
    'gender':[gen],
    'age':[age],
    'parental_education_level':[parent_edu],
    'family_income':[family_income],
    'daily_study_hours':[study_time],
    'attendance_rate':[attendance/100],
    'sleep_hours':[sleep_time],
    'stress_level':[stress],
    'motivation_score':[motivation],
    'private_tutoring':[private_tutor],
    'internet_quality':[internet_quality],
}) 
    prediction = deployed_model.predict(new_student)
    st.markdown('---')

    display_name=nam if nam else 'This Student'

    if prediction[0] == 1:

        st.success(f"✅ **PREDICTION: PASS** -{display_name} is on track!")
        st.balloons()
    else:
        st.error(f"⚠️ **PREDICTION: FAIL** - {display_name} requires some guidance.")