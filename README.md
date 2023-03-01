# BTC-predict

#Predict the bitcoin price using Coinbase API and predict the trading trend using Python 



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import requests
import csv

# Set up the API endpoint and parameters
url = "https://api.coinbase.com/v2/prices/BTC-USD/historic"
headers = {
    "Authorization": "Bearer <your_api_key>",
    "Accept": "application/json",
}

# Set up the API request parameters
params = {
    "period": "day",
    "start": "2022-02-01T00:00:00Z",
    "end": "2022-02-28T23:59:59Z",
    "granularity": 86400  # 1 day in seconds
}

# Send the API request and retrieve the response
response = requests.get(url, headers=headers, params=params)
data = response.json()

# Extract the Bitcoin price data from the response
prices = [[row["time"], row["price"]] for row in data["data"]["prices"]]

# Write the Bitcoin price data to a CSV file
with open("bitcoin_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time", "price"])
    writer.writerows(prices)

# Load the Bitcoin price data into a pandas DataFrame
bitcoin_df = pd.read_csv("bitcoin_data.csv")

# Split the data into training and testing sets
X = bitcoin_df[["time"]]
y = bitcoin_df[["price"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the performance of the model using the mean squared error metric
mse = ((y_pred - y_test) ** 2).mean()
print("Mean Squared Error: {:.2f}".format(mse))







