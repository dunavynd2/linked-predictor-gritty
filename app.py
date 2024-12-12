import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# Page config
st.set_page_config(page_title='LinkedIn User Predictor', layout='wide')

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('models/linkedin_model.joblib')

model = load_model()

# Title and description
st.title('LinkedIn User Predictor')

# Income options (1-9)
income_options = {
    "Less than $10,000": 1,
    "$10,000 to $19,999": 2,
    "$20,000 to $29,999": 3,
    "$30,000 to $39,999": 4,
    "$40,000 to $49,999": 5,
    "$50,000 to $74,999": 6,
    "$75,000 to $99,999": 7,
    "$100,000 to $150,000": 8,
    "Greater than $150,000": 9,
}

# Education options (1-8)
education_options = {
     "Less than high school": 1,
    "High school incomplete": 2,
    "High school graduate": 3,
    "Some college, no degree": 4,
    "Two-year associate degree": 5,
    "Four-year college/university degree": 6,
    "Some postgraduate or professional schooling": 7,
    "Postgraduate or professional degree": 8
}

# Create input form
with st.form("prediction_form"):
    st.subheader("Enter Demographic Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.selectbox('Income Level (1-9)', 
                            options=list(income_options.keys()))
        
        education = st.selectbox('Education Level (1-8)', 
                               options=list(education_options.keys()))
        
        age = st.slider('Age (18-98)', min_value=18, max_value=98, value=30)
    
    with col2:
        parent = st.radio('Parent', ['No', 'Yes'])
        married = st.radio('Married', ['No', 'Yes'])
        female = st.radio('Female', ['No', 'Yes'])

    submit_button = st.form_submit_button(label='Predict LinkedIn Usage')

if submit_button:
    # Convert selections to numeric values
    income_value = income_options[income]
    education_value = education_options[education]
    
    # Prepare features
    features = pd.DataFrame({
        'income': [income_value],
        'education': [education_value],
        'parent': [1 if parent == 'Yes' else 0],
        'marital': [1 if married == 'Yes' else 0],
        'age': [age],
        'female': [1 if female == 'Yes' else 0]
    })

    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    
 # Create visualization data
    prob_data = pd.DataFrame({
        'Outcome': ['Not LinkedIn User', 'LinkedIn User'],
        'Probability': [1 - probability, probability]
    })

    # Create Altair chart
    chart = alt.Chart(prob_data).mark_bar().encode(
        x=alt.X('Outcome:N', title='Prediction'),
        y=alt.Y('Probability:Q', 
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format='%')),
        color=alt.Color('Outcome:N',
                       scale=alt.Scale(domain=['LinkedIn User', 'Not LinkedIn User'],
                                     range=['#1f77b4', '#ff7f0e']))
    ).properties(
        title='Prediction Probabilities',
        width=400,
        height=300
    )
    # Show results
    st.header('Prediction Results')
    
    # Display prediction
    result = 'LinkedIn User' if prediction[0] == 1 else 'Not a LinkedIn User'
    prob_pct = probability * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Category", result)
        st.metric("Probability of being a LinkedIn User", f"{prob_pct:.1f}%")
    with col2:
        st.altair_chart(chart, use_container_width=True)
