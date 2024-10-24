import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_path = "data1.csv"
data = pd.read_csv(file_path)

wheat_data = data[(data['Item'] == 'Wheat and products') & (data['Element'] == 'Production')]
wheat_data = wheat_data[['Year', 'Value']].dropna()

X = wheat_data[['Year']]
y = wheat_data['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Root Mean Squared Error (RMSE): {rmse}")
print("Sample predictions for test set:", y_pred[:5])

user_year = int(input("Enter a year to predict wheat production: "))
future_years = pd.DataFrame({'Year': [user_year]})
future_years_scaled = scaler.transform(future_years)
future_prediction = model.predict(future_years_scaled)

print(f"Predicted wheat production for {user_year}: {future_prediction[0]}")

plt.figure(figsize=(18, 18))

# Line Plot
plt.subplot(3, 1, 1)
plt.plot(wheat_data['Year'], wheat_data['Value'], marker='o', linestyle='-', color='blue')
plt.title('Wheat Production Over Years')
plt.xlabel('Year')
plt.ylabel('Production Value (in tons)')
plt.grid(True)

# Histogram
plt.subplot(3, 1, 2)
plt.hist(wheat_data['Value'], bins=10, color='purple', alpha=0.7)
plt.title('Distribution of Wheat Production Values')
plt.xlabel('Production Value (in tons)')
plt.ylabel('Frequency')
plt.grid(axis='y')

# Cumulative Production Plot
plt.subplot(3, 1, 3)
plt.fill_between(wheat_data['Year'], wheat_data['Value'], color='skyblue', alpha=0.5)
plt.plot(wheat_data['Year'], wheat_data['Value'], color='Slateblue', alpha=0.6)
plt.axvline(user_year, color='red', linestyle='--', label='Predicted Year')
plt.scatter([user_year], future_prediction, color='green', s=100, label='Predicted Production')
plt.title('Cumulative Wheat Production Over Years')
plt.xlabel('Year')
plt.ylabel('Production Value (in tons)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()