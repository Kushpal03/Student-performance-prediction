import streamlit as st
import pandas as pd
import time
import joblib

model=joblib.load('model.pkl')
st.set_page_config(page_title='Bulk Predict',initial_sidebar_state='collapsed')

template_data = pd.DataFrame({
        'Name' : ['Example Student'],
        'gender':['Male'],
        'age':[15],
        'parental_education_level':[5],
        'family_income':[100000],
        'daily_study_hours':[5.5],
        'attendance_rate':[0.85],
        'sleep_hours':[8.5],
        'stress_level':[2],
        'motivation_score':[78],
        'private_tutoring':[True],
        'internet_quality':[5]
    })

csv_template = template_data.to_csv(index=False).encode('utf-8') 
col1,col2 = st.columns(2,gap='small')

with col1:

    if st.button('🛖 Back to Home',use_container_width=True):
        st.switch_page("deploy.py")
with col2:
    st.download_button(label='⬇️ Download Blank CSV Template',use_container_width=True,
                       file_name='student_prediction_template.csv',
                       data=csv_template,
                       mime='text/csv'
                       )

with st.expander('📝 Click here to view CSV formatting rules.'):
    st.write('**Please ensure your uploaded CSV has these exact columns :**')
    rul_col1,rul_col2 = st.columns(2)
    with rul_col1:
        
        st.markdown("""
                    * **gender:** Must be typed exactly as Male or Female.
                    * **age:** Must be an integer.
                    * **parental_education_level:** Must be a whole number between 1 and 9.
                    * **family_income:** Must be a whole number.
                    * **daily_study_hours:** Must be a decimal upto 1 decimal place.
                    * **attendance_rate:** Must be between 0 and 1.
                    """)
    with rul_col2:
        st.markdown("""
                    * **stress_level:** Must be an integer between 0 and 10.
                    * **motivation_score:** Must be whole number between 0 and 100.
                    * **sleep_hours:** Must be a decimal upto 1 decimal place.
                    * **internet_quality:** Must be a whole number between 1 and 5.
                    * **private_tutoring:** Must be typed exactly as True or False.
                    """)
    st.caption('*(Note: Any additional columns present in your file need not be removed they can be uploaded as is .The system automatically ignores any additional columns.)*')
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
            
            res_col1,res_col2=st.columns(2,gap='small',vertical_alignment='bottom')

            with res_col1:

                res_file_name = st.text_input('Enter the Result File Name',value='Class Results')
                
                if not res_file_name.endswith('.csv'):
                    final_file_name = res_file_name + '.csv'
                else:
                    final_file_name = res_file_name
            with res_col2:
                st.download_button(label='⬇️ Download the Results for the Class',
                                data=csv_results ,
                                file_name=final_file_name,
                                mime='text/csv',use_container_width=True)
