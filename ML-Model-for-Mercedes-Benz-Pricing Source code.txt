import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

# Load the dataset
try:
    df = pd.read_excel('D:/usa_mercedes_benz_prices.xlsx')
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error: {e}")
    exit()

# Check if required columns exist in the DataFrame
required_columns = ['Name', 'Rating', 'Review Count', 'Price']
if not all(column in df.columns for column in required_columns):
    print("Error: One or more required columns ('Name', 'Rating', 'Review Count', 'Price') not found in the dataset.")
    exit()

# Extract 'Model' and 'Year' from 'Name' column
df['Model'] = df['Name'].apply(lambda x: x.split(' ')[0])  # Assuming the model name is the first word
df['Year'] = df['Name'].str.extract('(\d{4})')

# Convert 'Year' to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Remove rows with missing 'Year' values
df = df.dropna(subset=['Year'])

# Explore distributions of numerical features
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(df['Rating'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')

plt.subplot(1, 2, 2)
plt.hist(df['Review Count'], bins=20, color='salmon', edgecolor='black')
plt.xlabel('Review Count')
plt.ylabel('Frequency')
plt.title('Distribution of Review Counts')

plt.tight_layout()
plt.show()

# Feature Engineering
df['Name_Length'] = df['Name'].apply(lambda x: len(x))
df['Brand'] = df['Name'].apply(lambda x: x.split(' ')[1])  # Assuming brand is the second word

# Check if required columns exist in the DataFrame
required_columns = ['Name', 'Rating', 'Review Count', 'Price', 'Model']
if not all(column in df.columns for column in required_columns):
    print("Error: One or more required columns ('Name', 'Rating', 'Review Count', 'Price', 'Model') not found in the dataset.")
    exit()

# Preprocess 'Price' column: remove dollar sign and commas
df['Price'] = df['Price'].replace('[\$,]', '', regex=True)

# Convert 'Price' to numeric, ignoring 'Not Priced' values
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Remove rows with missing or 'Not Priced' 'Price' values
df = df.dropna(subset=['Price'])

# Prepare the data for classification
X_classification = df[['Rating', 'Review Count', 'Year', 'Model']]
y_classification = df['Model']

# Split the data into training and testing sets for classification
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42)

# Train the Random Forest classifier
classification_model = RandomForestClassifier(random_state=42)
classification_model.fit(X_train_classification, y_train_classification)

# Predict on the test set for classification
y_pred_classification = classification_model.predict(X_test_classification)

# Evaluate the classification model
accuracy_classification = accuracy_score(y_test_classification, y_pred_classification)
print(f"Accuracy: {accuracy_classification:.2f}")
print("Classification Report:")
print(classification_report(y_test_classification, y_pred_classification, zero_division=1))

# Prepare the data for regression
X_regression = df[['Year']]
y_regression = df['Price']

# Split the data into training and testing sets for regression
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42)

# Train the Random Forest regressor
regression_model = RandomForestRegressor(random_state=42)

# Check if 'Model' column is present before one-hot encoding
if 'Model' in X_train_regression.columns:
    X_train_regression.drop('Model', axis=1, inplace=True)
if 'Model' in X_test_regression.columns:
    X_test_regression.drop('Model', axis=1, inplace=True)

# Fit the regression model
regression_model.fit(X_train_regression, y_train_regression)

# Predict the prices on the test set for regression
y_pred_regression = regression_model.predict(X_test_regression)

# Calculate the mean squared error of the predictions
mse_regression = mean_squared_error(y_test_regression, y_pred_regression)
print(f"Mean Squared Error: {mse_regression}")

# Analyze errors in regression predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test_regression, y_pred_regression, alpha=0.5)
plt.plot([min(y_test_regression), max(y_test_regression)], [min(y_test_regression), max(y_test_regression)], color='red', linestyle='--')
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.title('Regression Model: True vs Predicted Prices')
plt.show()

# Predict prices for the years 2025 to 2030
future_years = pd.DataFrame({'Year': range(2025, 2031)})
future_prices = regression_model.predict(future_years)
future_years['PredictedPrice'] = future_prices

# Display the predicted prices for future years
print("Predicted Prices for 2025-2030:")
print(future_years)

# Plot the predicted prices for future years
plt.figure(figsize=(10, 6))
plt.plot(future_years['Year'], future_years['PredictedPrice'], marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Predicted Price')
plt.title('Predicted Prices for 2025-2030')
plt.grid(True)
plt.show()
