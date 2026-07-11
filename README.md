# Retail Sales Time Series Forecasting

This project builds a machine learning pipeline to forecast weekly department sales across multiple retail stores. It includes the complete data preprocessing, feature engineering, and model validation pipeline, comparing gradient boosting methods with deep learning recurrent networks.

## Final Model Performance

The final champion model selected for production is a Tuned XGBoost Regressor. It was validated using a strict, chronological time-series split to ensure realistic evaluation.

* Validation Mean Absolute Error (MAE): $1,547.79
* Validation Root Mean Squared Error (RMSE): $3,160.38
* R-squared Score: 0.9794

## Core Technical Insights

### Validation Strategy
Standard random cross-validation cannot be used on this data because shuffling records shuffles time. This causes data leakage, where the model accidentally trains on future numbers to predict the past. To solve this, we used an expanding window TimeSeriesSplit, which trains strictly on past data and tests on the subsequent future window.

### Feature Power
Looking at the feature importances of our champion model, the single most powerful variable is our engineered lag feature, Sales_Lag_1, which accounts for 69.2% of the total predictive weight. This proves that the previous week's sales are the strongest anchor for forecasting future demand. We also created a Size_x_Holiday interaction feature to accurately capture how larger stores experience massive sales spikes during holiday weeks compared to smaller ones.

### XGBoost vs Deep Learning
While neural networks are popular for sequential data, the GRU network failed to converge on this problem, flatlining with a negative R-squared score. This happened because retail data combines hundreds of different stores and departments into a single tabular file, creating a jumbled sequence. The sharp, explosive holiday sales spikes caused gradient saturation in the GRU, forcing it to predict a simple rolling average. XGBoost handled this tabular structure much better, creating sharp decision boundaries that isolated holiday spikes and kept errors low.

## Repository Contents

* WalmartWeeklySalesPrediction.ipynb: The complete Jupyter Notebook containing all data cleaning, feature engineering, model training, and evaluation code.
* final_submission.csv: The final model predictions, formatted exactly to match the project's submission template.
* requirements.txt: The exact package versions required to reproduce this environment.
* README.md: This project overview document.
