# IPL-Score-Predictor

# Project Overview
This project aims to predict the first innings score in an Indian Premier League (IPL) match based on various factors such as team performance, venue, and match conditions. The model is built using machine learning techniques to estimate a reasonable target score.

# Dataset
The dataset contains past IPL match records.
It includes features like:
Batting team
Bowling team
Venue
Current score
Overs bowled
Wickets lost
Run rate
Last five overs' performance

# Libraries Used
The following Python libraries are used in this notebook:

pandas – for data manipulation
numpy – for numerical operations
matplotlib & seaborn – for data visualization
scikit-learn – for building and evaluating the predictive model

# Steps in the Notebook
Data Loading
The dataset is imported into a pandas DataFrame.
Data Preprocessing
Handling missing values (if any).
Encoding categorical variables (e.g., team names, venues).
Feature selection for model training.
Exploratory Data Analysis (EDA)
Visualizing trends and correlations.
Analyzing team performance over time.
Model Training
Regression models such as Linear Regression, Random Forest, or XGBoost are used.
The dataset is split into training and test sets.

# Model Evaluation
Performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score are used to assess accuracy.
Prediction & Conclusion
The trained model is used to predict the first innings score based on live inputs.
Insights and possible improvements are discussed.
