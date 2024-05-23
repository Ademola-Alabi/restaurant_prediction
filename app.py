import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('random_forest_regressor_model.joblib')
scaler = joblib.load('scaler.joblib')

# Function to make predictions
def predict(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]

# Streamlit app
st.set_page_config(page_title="Restaurant Income Prediction", page_icon="🍽️")


# Title and Subheader
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Restaurant Monthly Income Prediction Model</h1>", unsafe_allow_html=True)

# Display an image
st.image('restaurant_image.jpg', use_column_width=True)  # Replace with your image URL

st.markdown("<h3 style='text-align: center; color: #FF5733;'>Predict the potential monthly income based on various features of your restaurant</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #2C3E50;'>Enter the details below to get an estimate of your restaurant's monthly income. This model takes into account factors such as the number of customers, menu pricing, marketing expenditure, and more.</p>", unsafe_allow_html=True)


# Define input fields
st.markdown("<h4 style='color: #1E8449;'>Number of Customers:</h4>", unsafe_allow_html=True)
number_of_customers = st.number_input('', min_value=0, key='num_customers', help='Enter the number of customers visiting the restaurant.')

st.markdown("<h4 style='color: #1E8449;'>Menu Price:</h4>", unsafe_allow_html=True)
menu_price = st.number_input('', min_value=0.0, key='menu_price', help='Enter the average price of items on the menu.')

st.markdown("<h4 style='color: #1E8449;'>Marketing Spent:</h4>", unsafe_allow_html=True)
marketing_spent = st.number_input('', min_value=0.0, key='marketing_spent', help='Enter the amount spent on marketing.')

st.markdown("<h4 style='color: #1E8449;'>Average Customer Spending:</h4>", unsafe_allow_html=True)
average_customer_spending = st.number_input('', min_value=0.0, key='avg_customer_spending', help='Enter the average amount spent by each customer.')

st.markdown("<h4 style='color: #1E8449;'>Promotions:</h4>", unsafe_allow_html=True)
promotions = st.selectbox('', ['No Promotion', 'Promotion'], key='promotions', help='Select whether there is an ongoing promotion.')

st.markdown("<h4 style='color: #1E8449;'>Reviews:</h4>", unsafe_allow_html=True)
reviews = st.number_input('', min_value=0, key='reviews', help='Enter the number of reviews received.')

st.markdown("<h4 style='color: #1E8449;'>Cuisine:</h4>", unsafe_allow_html=True)
cuisine = st.selectbox('', ['American', 'Italian', 'Japanese', 'Mexican'], key='cuisine', help='Select the type of cuisine offered.')

# Convert categorical data to numeric
promotions = 1 if promotions == 'Promotion' else 0
cuisine_dict = {'American': 0, 'Italian': 1, 'Japanese': 2, 'Mexican': 3}
cuisine = cuisine_dict[cuisine]

# Collect user input into a single array
features = [number_of_customers, menu_price, marketing_spent, average_customer_spending, promotions, reviews, cuisine]

# Predict button
if st.button('Predict', key='predict_button', help='Click to predict the monthly income'):
    prediction = predict(features)
    st.markdown(f"<h3 style='text-align: center; color: #2980B9;'>The predicted monthly income is: ${prediction:.2f}</h3>", unsafe_allow_html=True)
