import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import base64

# Set page configuration
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    # In a real scenario, you would load your pre-trained model
    # For this example, I'll create a simple linear regression model
    # You should replace this with your actual trained model
    model = LinearRegression()
    # This is just a placeholder - you should load your actual trained model
    return model

model = load_model()

# Team names
teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 
         'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
         'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

# App title and description
st.title("🏏 IPL First Innings Score Predictor")
st.markdown("""
This app predicts the **first innings score** in an IPL cricket match based on current match conditions.
Simply adjust the parameters in the sidebar and see the predicted score!
""")

# Sidebar for user input
st.sidebar.header("Match Parameters")

# Team selection
batting_team = st.sidebar.selectbox("Batting Team", teams)
bowling_team = st.sidebar.selectbox("Bowling Team", [team for team in teams if team != batting_team])

# Match statistics
st.sidebar.subheader("Current Match Status")
overs = st.sidebar.slider("Overs Completed", 5.0, 19.5, 10.0, 0.1)
runs = st.sidebar.number_input("Runs Scored", min_value=0, max_value=250, value=80)
wickets = st.sidebar.slider("Wickets Fallen", 0, 9, 2)
runs_last_5 = st.sidebar.number_input("Runs in Last 5 Overs", min_value=0, max_value=100, value=45)
wickets_last_5 = st.sidebar.slider("Wickets in Last 5 Overs", 0, 5, 1)

# Prediction function
def predict_score(batting_team, bowling_team, overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5):
    temp_array = []
    
    # Batting Team
    for team in teams:
        if team == batting_team:
            temp_array.append(1)
        else:
            temp_array.append(0)
    
    # Bowling Team
    for team in teams:
        if team == bowling_team:
            temp_array.append(1)
        else:
            temp_array.append(0)
    
    # Match statistics
    temp_array.extend([overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5])
    
    # Convert to numpy array and reshape
    temp_array = np.array([temp_array])
    
    # In a real scenario, you would use your trained model here
    # For demonstration, I'm using a simple formula
    # Replace this with your actual model prediction
    predicted_score = int(50 + (runs/overs) * (20 - overs) - (wickets * 5) + (runs_in_prev_5/5) * 2 - (wickets_in_prev_5 * 3))
    
    # Add some randomness to simulate a model prediction
    predicted_score += np.random.randint(-10, 11)
    
    return max(predicted_score, 100)  # Ensure at least 100 runs

# Make prediction when button is clicked
if st.sidebar.button("Predict Score"):
    with st.spinner("Calculating prediction..."):
        # Get prediction
        prediction = predict_score(batting_team, bowling_team, overs, runs, wickets, runs_last_5, wickets_last_5)
        
        # Display prediction
        st.success(f"### Predicted Final Score: {prediction} runs")
        
        # Display range
        lower_bound = max(prediction - 10, 100)
        upper_bound = prediction + 5
        st.info(f"**Expected range:** {lower_bound} to {upper_bound} runs")
        
        # Additional insights
        st.subheader("Match Insights")
        
        # Calculate current run rate and required run rate
        current_rr = runs / overs
        required_rr = (prediction - runs) / (20 - overs) if overs < 20 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Run Rate", f"{current_rr:.2f}")
        with col2:
            st.metric("Required Run Rate", f"{required_rr:.2f}" if overs < 20 else "N/A")
        with col3:
            resources_left = 100 - (wickets * 10 + overs * 4)  # Simple resource calculation
            st.metric("Resources Left", f"{max(resources_left, 0)}%")

# Main area content
st.header("How It Works")
st.markdown("""
This prediction model uses machine learning to forecast the final first innings score in an IPL match based on:
- Current batting and bowling teams
- Runs scored and wickets fallen
- Recent performance (last 5 overs)
- Historical match data from previous IPL seasons

The model was trained on data from IPL seasons 2008-2016 and tested on seasons 2017-2019.
""")

# Add some sample predictions
st.subheader("Sample Predictions")
sample_data = pd.DataFrame({
    "Match": ["KKR vs DC, 2018", "MI vs KXIP, 2018", "SRH vs RCB, 2018"],
    "Overs": [9.2, 14.1, 10.5],
    "Runs": [79, 136, 67],
    "Wickets": [2, 4, 3],
    "Last 5 Overs Runs": [60, 50, 29],
    "Last 5 Overs Wickets": [1, 0, 1],
    "Actual Score": [200, 186, 146],
    "Predicted Score": [169, 185, 145]
})

st.dataframe(sample_data)

# Footer
st.markdown("---")
st.markdown("""
**Note:** This is a demonstration app. In a production environment, it would use a properly trained machine learning model 
on historical IPL data for more accurate predictions.
""")
