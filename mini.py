
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train:\n", X_train)
print("\nX_test:\n", X_test)
print("\ny_train:\n", y_train)
print("\ny_test:\n", y_test)


# Sample dataset creation (or load your CSV file here)
data = {
    'bedrooms': [2, 3, 4, 3, 5, 4, 2],
    'bathrooms': [1, 2, 2, 2, 3, 2, 1],
    'sqft': [1000, 1500, 2000, 1600, 2500, 2200, 1200],
    'age': [10, 5, 2, 8, 1, 3, 12],
    'price': [200000, 300000, 400000, 320000, 500000, 450000, 220000]
}

# Converting to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['bedrooms', 'bathrooms', 'sqft', 'age']]
y = df['price']

print("X data:\n",X)
print("\ny data:\n",y)
# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train:\n", X_train)
print("\nX_test:\n", X_test)
print("\ny_train:\n", y_train)


# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Creating a DataFrame to compare X_test and y_pred
results_df = X_test.copy()  # Copy X_test
results_df['Predicted Price'] = y_pred  # Add predictions as a new column

# Display the results
print("\nX_test with Predicted Prices:\n", results_df)
# Evaluating the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
# Displaying the coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print("Intercept:", model.intercept_)
print("\ny_test:\n", y_test)
