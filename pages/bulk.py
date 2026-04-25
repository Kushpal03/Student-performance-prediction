import streamlit as st
import pandas as pd
import time
import joblib

model=joblib.load('model.pkl')
st.set_page_config(page_title='Bulk Predict',initial_sidebar_state='collapsed')

if st.button('🛖 Back to Home'):
    st.switch_page("deploy.py")

@st.cache_data
def load_data(file):
    time.sleep(1)
    return pd.read_csv(file)

st.title('📁 Bulk Student Predictor')
st.write('Upload a CSV file containing your class roster to predict all outcomes at once.')


uploaded_file = st.file_uploader('Upload CSV' , type=['csv'])

if uploaded_file is not None:
    with st.spinner('Uploading File...'):
        
        bulk_data = load_data(uploaded_file)
    predict_data = bulk_data.copy()

    expected_col = [
        'gender',
        'age',
        'parental_education_level',
        'family_income',
        'daily_study_hours',
        'attendance_rate',
        'sleep_hours',
        'stress_level',
        'motivation_score',
        'private_tutoring',
        'internet_quality']
        
        

        

    missing_col = [col for col in expected_col if col not in predict_data.columns]

    if len(missing_col) >0:
        st.error(f"⚠️Upload failed! Your CSV is missing the following required columns : {','.join(missing_col)}")
        st.stop()
    if st.button('Predict all Students'):
         with st.spinner('Processing Roster...'):
            time.sleep(1)

            predict_data = bulk_data[expected_col]

            predictions = model.predict(predict_data)
            final_results = bulk_data.copy()
            final_results['Predicted Outcome'] = predictions
            final_results['Predicted Outcome'] = final_results['Predicted Outcome'].map({1 : 'Pass' , 0 : 'Fail'})
            st.success('Predictions Complete')
            st.dataframe(final_results.head(10))
            st.balloons()

            csv_results = final_results.to_csv(index=False).encode('utf-8')
            

            
            res_file_name = st.text_input('Enter the Result File Name',value='Class Results')
            
            if not res_file_name.endswith('.csv'):
                final_file_name = res_file_name + '.csv'
            else:
                final_file_name = res_file_name

            st.download_button(label='⬇️ Download the Results for the Class',
                            data=csv_results ,
                            file_name=final_file_name,
                            mime='text/csv')
