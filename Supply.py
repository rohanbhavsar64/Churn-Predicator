import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
order=pd.read_csv('order (1).csv')

st.write(order)
order['Order Date'] = order['Order Year'].astype(str) + '/' + order['Order Month'].astype(str) + '/' + order['Order Day'].astype(str)
order['Order Date'] = pd.to_datetime(order['Order Date'])
order['Order Date'] = order['Order Date'].dt.strftime('%Y/%m/%d')
order['Order YearMonth'] = order['Order YearMonth'].astype(str)
data = order
# Extract year and demand
data['Order Date']=pd.to_datetime(data['Order Date'])
data['Year'] = data['Order Date'].dt.year
data['Day of Year'] = data['Order Date'].dt.dayofyear
data = data.groupby(['Year', 'Day of Year'])['Order Quantity'].sum().reset_index()

# Prepare training data
X = data[data['Year'] < 2018][['Year', 'Day of Year']]
y = data[data['Year'] < 2018]['Order Quantity']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=200,max_depth=7, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on test data
y_pred_test = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)

# Display evaluation metrics
print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Predict for 2015–2017
X_2015_2017 = data[data['Year'] < 2018][['Year', 'Day of Year']]
y_actual_2015_2017 = data[data['Year'] < 2018]['Order Quantity']
y_pred_2015_2017 = model.predict(X_2015_2017)

# Plot actual vs predicted for 2015–2017
plt.figure(figsize=(12, 6))
plt.plot(data[data['Year'] < 2018]['Day of Year'], y_actual_2015_2017, label='Actual Demand (2015–2017)', color='green', alpha=0.6)
plt.plot(data[data['Year'] < 2018]['Day of Year'], y_pred_2015_2017, label='Predicted Demand (2015–2017)', color='red', linestyle='--')
plt.xlabel('Day of Year')
plt.ylabel('Demand')
plt.title('Actual vs Predicted Demand (2015–2017)')
plt.legend()
plt.grid()
plt.show()

# Predict for 2018
X_2018 = data[data['Year'] == 2018][['Year', 'Day of Year']]
if X_2018.empty:  # Generate dates if 2018 data is absent
    days_2018 = pd.date_range('2018-01-01', '2018-12-31', freq='D').dayofyear
    X_2018 = pd.DataFrame({'Year': 2018, 'Day of Year': days_2018})

y_pred_2018 = model.predict(X_2018)
# Plot predicted demand for 2018
a=plt.figure(figsize=(12, 6))
plt.plot(X_2018['Day of Year'], y_pred_2018, label='Predicted Demand for 2018', color='blue')
plt.xlabel('Day of Year')
plt.ylabel('Demand')
plt.title('Demand Forecasting for 2018 (Gradient Boosting Regeressor)')
plt.legend()
plt.grid()
st.write(a)                        
