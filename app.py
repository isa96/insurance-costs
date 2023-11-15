
import streamlit as st
import numpy as np
import json
import joblib


def json_file():
	with open("features.json") as columns:
		data_json = json.loads(columns.read())
		data_json = np.asarray(data_json['features'])
	return data_json

st.cache(allow_output_mutation=True)

def input_data():
	region = st.radio("Region",('Southeast', 'Northeast', 'Southwest', 'Northwest'))
	age = st.slider(label="Age", min_value=18, max_value=64, step=1)
	sex = st.selectbox("Sex", ('Male', 'Female'))
	children = st.slider(label="Children", min_value=0, max_value=5, step=1)
	bmi = st.number_input(label="BMI", min_value=18.0, max_value=47.0, step=0.0001)
	medical_problem = st.selectbox("Medical Problem", ('Light', 'Severe'))
	smoker = st.selectbox("Smoker", ('No', 'Yes'))

	return region, age, sex, children, bmi, medical_problem, smoker


def preprocessing():
    
    region,age,sex,children,bmi,medical_problem,smoker = input_data()
    
    columns = json_file()
    
    new_data = np.zeros(len(columns))
    
    region_idx = np.where(region == columns)[0][0]
    
    
    if region_idx >= 0:
        new_data[region_idx] = 1
        
        
    new_data[4] = age
    new_data[5] = np.where(sex  == 'Male',1,0)
    new_data[6] = children
    new_data[7] = bmi
    new_data[8] = np.where(medical_problem == 'Severe',1,0)
    new_data[9] = np.where(smoker == 'Yes',1,0)
    
    
    
    return np.asarray([new_data])


def predict(new_data):
    
    model = joblib.load('final_model.pkl')
    
    return model.predict(new_data)



def main():
    
    st.write(""" # Predicted Insurence Price """)

    # st.image("""bg-insurance.jpg""")
  
    new_data = preprocessing()
    
    
    if st.button(label = 'Predict'):
        
        charges_pred =predict(new_data)
        
        st.success(f'The estimated health insurance charge is: $ {charges_pred} USD')


st.cache(allow_output_mutation=True)


if __name__ == '__main__':
    main()

