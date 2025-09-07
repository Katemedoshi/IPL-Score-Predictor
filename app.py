import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# Set page configuration
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .pulse-animation {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🏏 IPL First Innings Score Predictor</h1>', unsafe_allow_html=True)

# Team names
teams = [
    'Chennai Super Kings', 
    'Delhi Daredevils', 
    'Kings XI Punjab', 
    'Kolkata Knight Riders', 
    'Mumbai Indians', 
    'Rajasthan Royals',
    'Royal Challengers Bangalore', 
    'Deccan Chargers'
]

# Venues
venues = [
    "M Chinnaswamy Stadium", 
    "Punjab Cricket Association Stadium, Mohali",
    "Feroz Shah Kotla", 
    "Wankhede Stadium",
    "Eden Gardens", 
    "Sawai Mansingh Stadium",
    "Rajiv Gandhi International Stadium, Uppal",
    "MA Chidambaram Stadium, Chepauk"
]

# Load sample data for visualization
@st.cache_data
def load_sample_data():
    # Create sample match data for visualization
    np.random.seed(42)
    sample_matches = []
    for i in range(50):
        overs = np.random.uniform(5, 20)
        runs = int(np.random.normal(overs * 8, 10))
        wickets = min(int(np.random.normal(overs / 4, 1.5)), 9)
        runs_last_5 = int(np.random.normal(40, 10))
        wickets_last_5 = min(int(np.random.normal(1.5, 0.8)), 5)
        total = int(runs + (20 - overs) * (runs/overs) * 0.9 - wickets * 5 + np.random.normal(0, 10))
        
        sample_matches.append({
            'Overs': round(overs, 1),
            'Runs': runs,
            'Wickets': wickets,
            'Runs_Last_5': runs_last_5,
            'Wickets_Last_5': wickets_last_5,
            'Total_Score': total
        })
    
    return pd.DataFrame(sample_matches)

# Load sample data
sample_data = load_sample_data()

# Sidebar for user input
st.sidebar.header("⚙️ Match Parameters")

# Team selection
st.sidebar.subheader("🏟️ Teams")
batting_team = st.sidebar.selectbox("Batting Team", teams, index=0)
bowling_team = st.sidebar.selectbox("Bowling Team", 
                                  [team for team in teams if team != batting_team], 
                                  index=1)

# Venue selection
venue = st.sidebar.selectbox("Stadium", venues, index=0)

# Match statistics
st.sidebar.subheader("📊 Current Match Status")

overs = st.sidebar.slider("Overs Completed", 5.0, 19.5, 10.0, 0.1)
runs = st.sidebar.number_input("Runs Scored", min_value=0, max_value=300, value=80)
wickets = st.sidebar.slider("Wickets Fallen", 0, 9, 2)
runs_last_5 = st.sidebar.number_input("Runs in Last 5 Overs", min_value=0, max_value=100, value=45)
wickets_last_5 = st.sidebar.slider("Wickets in Last 5 Overs", 0, 5, 1)

# Simulate model training
@st.cache_resource
def create_model():
    # This would be replaced with your actual trained model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train on sample data
    X = sample_data[['Overs', 'Runs', 'Wickets', 'Runs_Last_5', 'Wickets_Last_5']]
    y = sample_data['Total_Score']
    model.fit(X, y)
    
    return model

model = create_model()

# Prediction function
def predict_score(batting_team, bowling_team, venue, overs, runs, wickets, runs_last_5, wickets_last_5):
    
    # Prepare input features
    input_features = pd.DataFrame({
        'Overs': [overs],
        'Runs': [runs],
        'Wickets': [wickets],
        'Runs_Last_5': [runs_last_5],
        'Wickets_Last_5': [wickets_last_5]
    })
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Add some randomness to simulate real-world uncertainty
    uncertainty = np.random.normal(0, 5)
    predicted_score = int(prediction + uncertainty)
    
    # Ensure realistic score
    min_score = runs + int((20 - overs) * 4)  # At least 4 runs per over remaining
    max_score = runs + int((20 - overs) * 12)  # At most 12 runs per over remaining
    
    predicted_score = max(min(predicted_score, max_score), min_score)
    
    return predicted_score

# Calculate additional metrics
def calculate_metrics(overs, runs, wickets, predicted_score):
    # Current run rate
    current_rr = runs / overs if overs > 0 else 0
    
    # Required run rate
    remaining_overs = 20 - overs
    required_runs = predicted_score - runs
    required_rr = required_runs / remaining_overs if remaining_overs > 0 else 0
    
    # Projected score based on current RR
    projected_current_rr = runs / overs * 20 if overs > 0 else 0
    
    return {
        'current_rr': current_rr,
        'required_rr': required_rr,
        'projected_current_rr': projected_current_rr
    }

# Make prediction when button is clicked
if st.sidebar.button("🚀 Predict Score", use_container_width=True):
    with st.spinner("Calculating prediction..."):
        # Simulate processing time
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        # Get prediction
        prediction = predict_score(
            batting_team, bowling_team, venue, overs, runs, wickets, 
            runs_last_5, wickets_last_5
        )
        
        # Calculate metrics
        metrics = calculate_metrics(overs, runs, wickets, prediction)
        
        # Display prediction
        st.markdown("---")
        st.markdown('<div class="prediction-card pulse-animation">', unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #1E90FF;'>Predicted Final Score: {prediction} runs</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display range
        lower_bound = max(prediction - 10, runs + int((20 - overs) * 4))
        upper_bound = min(prediction + 10, runs + int((20 - overs) * 12))
        st.info(f"**Expected range:** {lower_bound} to {upper_bound} runs")
        
        # Key metrics
        st.subheader("📈 Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Current Run Rate", f"{metrics['current_rr']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Required Run Rate", f"{metrics['required_rr']:.2f}" if overs < 20 else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Projected at Current RR", f"{metrics['projected_current_rr']:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.subheader("📊 Performance Analysis")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Run Rate Analysis", "Historical Comparison"])
        
        with tab1:
            # Run rate chart
            fig_rr = go.Figure()
            
            # Current run rate
            fig_rr.add_trace(go.Scatter(
                x=[overs], y=[metrics['current_rr']],
                mode='markers+text',
                marker=dict(size=15, color='red'),
                text=["Current RR"],
                textposition="top center",
                name="Current RR"
            ))
            
            # Required run rate
            if overs < 20:
                fig_rr.add_trace(go.Scatter(
                    x=[overs], y=[metrics['required_rr']],
                    mode='markers+text',
                    marker=dict(size=15, color='green'),
                    text=["Required RR"],
                    textposition="top center",
                    name="Required RR"
                ))
            
            # Projected run rate
            fig_rr.add_hline(y=metrics['current_rr'], line_dash="dash", line_color="red",
                            annotation_text="Current Run Rate", annotation_position="bottom right")
            
            if overs < 20:
                fig_rr.add_hline(y=metrics['required_rr'], line_dash="dash", line_color="green",
                                annotation_text="Required Run Rate", annotation_position="top right")
            
            fig_rr.update_layout(
                title="Run Rate Analysis",
                xaxis_title="Overs",
                yaxis_title="Run Rate",
                showlegend=True
            )
            
            st.plotly_chart(fig_rr, use_container_width=True)
        
        with tab2:
            # Historical comparison
            fig_hist = px.histogram(sample_data, x="Total_Score", 
                                   title="Distribution of Historical Scores in Similar Conditions")
            fig_hist.add_vline(x=prediction, line_dash="dash", line_color="red",
                              annotation_text="Predicted Score", annotation_position="top right")
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Match insights
        st.subheader("💡 Match Insights")
        
        # Generate insights based on the match situation
        insights = []
        
        if runs_last_5 > 50:
            insights.append("✅ Batting team has strong momentum with high runs in last 5 overs")
        elif runs_last_5 < 30:
            insights.append("⚠️ Batting team is struggling with low runs in last 5 overs")
        
        if wickets_last_5 >= 2:
            insights.append("⚠️ Bowling team has taken multiple wickets recently")
        
        if metrics['required_rr'] > 10 and overs < 15:
            insights.append("🎯 Batting team needs to accelerate significantly")
        elif metrics['required_rr'] < 8:
            insights.append("✅ Batting team is in a comfortable position")
        
        for insight in insights:
            st.info(insight)

# Main content when no prediction has been made yet
else:
    # Header
    st.markdown("""
    ## Welcome to the IPL Score Predictor!
    
    This tool uses machine learning to predict the first innings score in an IPL cricket match based on current match conditions.
    """)
    
    # How to use section
    st.subheader("📖 How to Use")
    
    st.markdown("""
    1. Select the batting and bowling teams from the sidebar
    2. Choose the venue where the match is being played
    3. Adjust the match parameters (overs, runs, wickets, etc.)
    4. Click the 'Predict Score' button to get your prediction
    5. Explore the detailed analytics and visualizations
    """)
    
    # Sample visualization
    st.subheader("📊 Run Rate vs Total Score (Sample Data)")
    
    fig_sample = px.scatter(sample_data, x="Overs", y="Runs", color="Total_Score",
                           size="Wickets", hover_data=["Runs_Last_5", "Wickets_Last_5"],
                           title="Relationship Between Overs, Runs, and Final Score")
    
    st.plotly_chart(fig_sample, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This is a demonstration application. Predictions are based on simulated data and machine learning models.
For accurate real-world predictions, more detailed historical data would be required.
""")
