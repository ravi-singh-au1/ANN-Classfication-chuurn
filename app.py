import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load preprocessing objects
with open('label_encoder_gender.pk1', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pk1', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pk1', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
secondcolumn=st.number_input('second column',value=0.0)
age = st.slider('Age', 18, 100)  # Changed max age from 20 to 100 for realism
balance = st.number_input('Balance', value=0.0)

# Prepare the input data
gender_encoded = label_encoder_gender.transform([gender])[0]

input_data = pd.DataFrame({
    'Gender': [gender_encoded],
    'Age': [age],
    'Balance': [balance],
    'second column':[secondcolumn]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())

# Concatenate all features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probality:{prediction_proba:.2f}')
# Display result
if prediction_proba > 0.5:
    st.success('⚠️ The customer is likely to churn.')
else:
    st.info('✅ The customer is not likely to churn.')
