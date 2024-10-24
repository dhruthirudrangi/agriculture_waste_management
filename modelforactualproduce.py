import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('production.csv')

data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

grouped_data = data.groupby(['Area', 'Item', 'Year'])['Value'].sum().reset_index()

user_area = input("Enter the Area you want to predict production for: ")
user_crop = input("Enter the Crop (Item) you want to predict production for: ")
user_year = int(input("Enter the year you want to predict production for: "))

filtered_data = grouped_data[(grouped_data['Area'] == user_area) & (grouped_data['Item'] == user_crop)]

X = filtered_data[['Year']]
y = filtered_data['Value']

if len(X) < 2:
    print("Not enough data for the specified area and crop.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    prediction = linear_model.predict(np.array([[user_year]]))

    print(f"Predicted Production Value for {user_area} ({user_crop}) in {user_year}: {prediction[0]}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Historical Data')
    plt.plot(X, linear_model.predict(X), color='orange', label='Linear Regression Line')
    plt.scatter(user_year, prediction, color='red', label='Predicted Value', s=100)
    plt.title(f'Production Prediction for {user_area} ({user_crop})')
    plt.xlabel('Year')
    plt.ylabel('Production Value')

    all_years = np.arange(int(X['Year'].min()), int(X['Year'].max()) + 2, 1)
    all_years = np.append(all_years, user_year)
    plt.xticks(ticks=np.unique(all_years))

    plt.axvline(x=user_year, color='red', linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()
