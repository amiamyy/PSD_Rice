import streamlit as st
import pickle
import numpy as np

# Load the trained Naive Bayes model from the pickle file
model_filename = 'naive_bayes_model.pkl'
with open(model_filename, 'rb') as file:
    clf = pickle.load(file)

# Function to make predictions
def predict_variety(area, major, minor, eccentricity, convex_area, extent, perimeter):
    features = np.array([[area, major, minor, eccentricity, convex_area, extent, perimeter]])
    prediction = clf.predict(features)
    return 'Osmancik' if prediction[0] == 1 else 'Cammeo'

# Streamlit app
st.title('Rice Variety Prediction')
st.write('Input the features to predict the rice variety (Osmancik or Cammeo).')

# Input fields for the features
area = st.number_input('Area', min_value=0.0, format="%.2f")
major = st.number_input('Major Axis Length', min_value=0.0, format="%.2f")
minor = st.number_input('Minor Axis Length', min_value=0.0, format="%.2f")
eccentricity = st.number_input('Eccentricity', min_value=0.0, format="%.2f")
convex_area = st.number_input('Convex Area', min_value=0.0, format="%.2f")
extent = st.number_input('Extent', min_value=0.0, format="%.2f")
perimeter = st.number_input('Perimeter', min_value=0.0, format="%.2f")

# Button to make predictions
if st.button('Predict'):
    result = predict_variety(area, major, minor, eccentricity, convex_area, extent, perimeter)
    st.write(f'The predicted rice variety is: *{result}*')

# To run this app, save it as a Python file (e.g., app.py) and run the following command in the terminal:
# streamlit run app.py